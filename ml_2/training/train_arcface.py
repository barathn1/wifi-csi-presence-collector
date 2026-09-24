"""Trains + evaluates the ArcFace open-set embedding (ml_2/models/arcface.py), on the SAME backbone
and SAME 3-tier split discipline as train_prototypical.py, so the two embedding-based candidates are
comparable to each other and not confounded by a different backbone. Same "no intruder data" training
philosophy: the backbone + ArcFace head only ever see known-identity (authorized) windows; the
open-set threshold is calibrated on a held-out slice of those, never on real strangers/none.

    python3 -m ml_2.training.train_arcface --epochs 8
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
from ml_2.data.metrics import compute_auroc, compute_eer
from ml_2.training.common_data import N_SUBCARRIERS as TARGET_SUBCARRIERS
from ml_2.models.arcface import ArcFaceHead
from ml_2.models.backbone import DualBranchEmbedding
from ml_2.training.gpu_utils import DEVICE, FOLD_DEVICES, run_folds_parallel
from ml_2.training.common_data import Channel6Dataset, load_channel6_dataset
from ml_2.training.train_prototypical import NON_AUTH_LABEL, split_known_train_calib

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/arcface_log.csv"
LOG_FIELDNAMES = ["timestamp", "split_type", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "calibrated_accept_rate",
                   "n_train", "n_test", "notes"]


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def train_backbone_and_head(backbone: DualBranchEmbedding, head: ArcFaceHead, centroid_ds: CsiWindowDataset,
                             global_to_local: dict[int, int], epochs: int, seed: int,
                             batch_size: int = 64, lr: float = 1e-3, device: torch.device = DEVICE) -> None:
    torch.manual_seed(seed)
    backbone.to(device)
    head.to(device)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = DataLoader(centroid_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, multiprocessing_context="fork")
    for _ in range(epochs):
        backbone.train()
        head.train()
        for amp, phase, label in loader:
            amp, phase = amp.to(device), phase.to(device)
            local_label = torch.tensor([global_to_local[int(l)] for l in label], device=device)
            emb = backbone(amp, phase)
            logits = head(emb, local_label)
            loss = loss_fn(logits, local_label)
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def score_all(backbone: DualBranchEmbedding, head: ArcFaceHead, ds, batch_size: int = 64,
              device: torch.device = DEVICE) -> np.ndarray:
    backbone.eval()
    head.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, multiprocessing_context="fork")
    scores = []
    for amp, phase, _ in loader:
        emb = backbone(amp.to(device), phase.to(device))
        scores.append(head.genuine_score(emb).cpu())
    return torch.cat(scores).numpy()


def run_fold(full_ds: CsiWindowDataset, train_idx: np.ndarray, test_idx: np.ndarray, epochs: int, seed: int,
             device: torch.device = DEVICE) -> dict:
    centroid_idx, calib_idx = split_known_train_calib(full_ds, train_idx, seed)
    known_class_ids = sorted(full_ds.class_to_idx[c] for c in full_ds.classes if c != NON_AUTH_LABEL)
    global_to_local = {g: i for i, g in enumerate(known_class_ids)}

    backbone = DualBranchEmbedding(n_subcarriers=TARGET_SUBCARRIERS)
    head = ArcFaceHead(embed_dim=32, n_classes=len(known_class_ids))
    centroid_ds = full_ds.subset_by_index_rows(centroid_idx)
    train_backbone_and_head(backbone, head, centroid_ds, global_to_local, epochs, seed, device=device)

    calib_scores = score_all(backbone, head, full_ds.subset_by_index_rows(calib_idx), device=device)
    threshold = float(np.percentile(calib_scores, 5))  # accepts ~95% of genuine calib windows

    test_scores = score_all(backbone, head, full_ds.subset_by_index_rows(test_idx), device=device)
    calibrated_accept = (test_scores >= threshold).astype(int)

    y_true = (full_ds.index.iloc[test_idx]["label"].values == "authorized").astype(int)
    eer, eer_threshold = compute_eer(y_true, test_scores)
    auroc = compute_auroc(y_true, test_scores)
    pred = (test_scores >= eer_threshold).astype(int) if not np.isnan(eer_threshold) else calibrated_accept
    out = {"accuracy": float((pred == y_true).mean()), "eer": eer, "auroc": auroc,
           "calibrated_accept_rate": float(calibrated_accept.mean()),
           "n_train": len(train_idx), "n_test": len(test_idx)}
    test_rows = full_ds.index.iloc[test_idx]
    original_labels = test_rows["label"].values
    for neg in ("unauthorized", "none"):
        m = original_labels == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())

    from ml_2.data.metrics import format_session_breakdown, session_level_metrics
    sm = session_level_metrics(y_true, test_scores, pred, test_rows["session_dir"].values,
                                test_rows["date"].values)
    out["session_accuracy"], out["session_auroc"], out["n_sessions"] = \
        sm["session_accuracy"], sm["session_auroc"], sm["n_sessions"]
    out["_session_breakdown_str"] = format_session_breakdown(sm)
    return out


def main(epochs: int, seed: int) -> None:
    dataset: Channel6Dataset = load_channel6_dataset()
    full_ds = CsiWindowDataset(dataset.window_index, "identity_or_nonauth", calibration=dataset.calibration)
    assert len(full_ds) == len(dataset.window_index), (len(full_ds), len(dataset.window_index))
    print(f"taskF classes: {full_ds.classes} ({len(full_ds)} windows, aligned 1:1 with window_index)")

    print(f"\n=== ArcFace open-set: leave-one-unauthorized-person-out "
          f"(parallel across {len(FOLD_DEVICES)} device(s)) ===")
    fold_specs = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(dataset.window_index, include_none=True):
        assert_no_group_leakage(dataset.window_index, train_idx, test_idx)
        fold_specs.append((held_out, train_idx, test_idx))
    fold_metrics = run_folds_parallel(
        fold_specs, lambda spec, device: {**run_fold(full_ds, spec[1], spec[2], epochs, seed, device=device),
                                            "held_out": spec[0]})
    for m in fold_metrics:
        print(f"  held out '{m['held_out']}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f} "
              f"calibrated_accept_rate={m['calibrated_accept_rate']:.3f} "
              f"false_accept_unauth={m.get('false_accept_unauthorized', float('nan')):.3f} "
              f"false_accept_none={m.get('false_accept_none', float('nan')):.3f}")
        print(m.pop("_session_breakdown_str", ""))
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "split_type": "open_set_loo_stranger", "notes": "leave-one-unauthorized-person-out"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")

    print(f"\n=== ArcFace cross-day: leave-one-day-out (parallel across {len(FOLD_DEVICES)} device(s)) ===")
    fold_specs = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(dataset.window_index):
        test_labels = dataset.window_index.iloc[test_idx]["label"]
        if (test_labels == "authorized").sum() == 0 or len(test_labels.unique()) < 2:
            print(f"  skipping {held_out_date}: degenerate class balance")
            continue
        fold_specs.append((held_out_date, train_idx, test_idx))
    fold_metrics = run_folds_parallel(
        fold_specs, lambda spec, device: {**run_fold(full_ds, spec[1], spec[2], epochs, seed, device=device),
                                            "held_out": spec[0]})
    for m in fold_metrics:
        print(f"  held out day '{m['held_out']}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f} "
              f"calibrated_accept_rate={m['calibrated_accept_rate']:.3f}")
        print(m.pop("_session_breakdown_str", ""))
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "split_type": "cross_day_loo", "notes": "leave-one-day-out"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")

    print(f"\nDone. Results -> {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
