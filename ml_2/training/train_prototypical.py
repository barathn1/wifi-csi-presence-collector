"""Trains + evaluates the prototypical-network open-set embedding (ml_2/models/prototypical.py +
backbone.py). Per CAUTION's own recipe: the backbone/centroids are trained using ONLY known-identity
(authorized) windows -- `non_auth` windows never participate in the loss -- and the intruder threshold
is calibrated afterward using a held-out slice of the SAME known-identity windows (never real
stranger/none data), then evaluated for real against genuinely unseen strangers and days.

    python3 -m ml_2.training.train_prototypical --epochs 8
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
from ml_2.models.backbone import DualBranchEmbedding
from ml_2.models.prototypical import compute_centroids, episodic_split, nearest_centroid_distance_ratio, prototypical_loss
from ml_2.training.common_data import Channel6Dataset, load_channel6_dataset
from ml_2.training.gpu_utils import DEVICE, FOLD_DEVICES, run_folds_parallel

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/prototypical_log.csv"
LOG_FIELDNAMES = ["timestamp", "split_type", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "calibrated_accept_rate",
                   "n_train", "n_test", "notes"]
NON_AUTH_LABEL = "non_auth"
CALIB_FRACTION = 0.2


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def split_known_train_calib(full_ds: CsiWindowDataset, train_idx: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions (into full_ds.index, aligned 1:1 with window_index since taskF's mask is all-True)
    of this fold's known-identity (non-`non_auth`) training rows, split into a centroid-fitting subset
    and a held-out calibration subset -- both still only ever see AUTHORIZED windows."""
    known_mask = full_ds.y[train_idx] != NON_AUTH_LABEL
    known_idx = train_idx[known_mask]
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(known_idx)
    n_calib = max(1, int(round(len(shuffled) * CALIB_FRACTION)))
    return shuffled[n_calib:], shuffled[:n_calib]


def train_backbone(backbone: DualBranchEmbedding, centroid_ds: CsiWindowDataset, known_class_ids: list[int],
                    epochs: int, seed: int, batch_size: int = 64, lr: float = 1e-3,
                    device: torch.device = DEVICE) -> None:
    torch.manual_seed(seed)
    backbone.to(device)
    opt = torch.optim.Adam(backbone.parameters(), lr=lr)
    loader = DataLoader(centroid_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, multiprocessing_context="fork")
    for _ in range(epochs):
        backbone.train()
        for amp, phase, label in loader:
            if len(torch.unique(label)) < 2:
                continue
            amp, phase, label = amp.to(device), phase.to(device), label.to(device)
            support_idx, query_idx = episodic_split(label, known_class_ids)
            emb = backbone(amp, phase)
            centroids = compute_centroids(emb[support_idx], label[support_idx], known_class_ids)
            loss = prototypical_loss(emb[query_idx], label[query_idx], centroids, known_class_ids)
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def embed_all(backbone: DualBranchEmbedding, ds, batch_size: int = 64,
              device: torch.device = DEVICE) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, multiprocessing_context="fork")
    embs, labels = [], []
    for amp, phase, label in loader:
        emb = backbone(amp.to(device), phase.to(device))
        embs.append(emb.cpu())
        labels.append(label)
    return torch.cat(embs), torch.cat(labels)


def run_fold(full_ds: CsiWindowDataset, train_idx: np.ndarray, test_idx: np.ndarray, epochs: int, seed: int,
             device: torch.device = DEVICE) -> dict:
    centroid_idx, calib_idx = split_known_train_calib(full_ds, train_idx, seed)
    known_class_ids = [full_ds.class_to_idx[c] for c in full_ds.classes if c != NON_AUTH_LABEL]

    backbone = DualBranchEmbedding(n_subcarriers=TARGET_SUBCARRIERS)
    centroid_ds = full_ds.subset_by_index_rows(centroid_idx)
    train_backbone(backbone, centroid_ds, known_class_ids, epochs, seed, device=device)

    centroid_embs, centroid_labels = embed_all(backbone, centroid_ds, device=device)
    centroids = compute_centroids(centroid_embs, centroid_labels, known_class_ids)

    calib_embs, _ = embed_all(backbone, full_ds.subset_by_index_rows(calib_idx), device=device)
    _, calib_ratio = nearest_centroid_distance_ratio(calib_embs, centroids)
    threshold = float(np.percentile(calib_ratio.numpy(), 95))  # accepts ~95% of genuine calib windows

    test_embs, _ = embed_all(backbone, full_ds.subset_by_index_rows(test_idx), device=device)
    nearest_idx, test_ratio = nearest_centroid_distance_ratio(test_embs, centroids)
    genuine_score = (-test_ratio).numpy()  # lower ratio = more confidently a known identity = more genuine
    calibrated_accept = (test_ratio.numpy() <= threshold).astype(int)

    y_true = (full_ds.index.iloc[test_idx]["label"].values == "authorized").astype(int)
    eer, eer_threshold = compute_eer(y_true, genuine_score)
    auroc = compute_auroc(y_true, genuine_score)
    pred = (genuine_score >= eer_threshold).astype(int) if not np.isnan(eer_threshold) else calibrated_accept
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
    sm = session_level_metrics(y_true, genuine_score, pred, test_rows["session_dir"].values,
                                test_rows["date"].values)
    out["session_accuracy"], out["session_auroc"], out["n_sessions"] = \
        sm["session_accuracy"], sm["session_auroc"], sm["n_sessions"]
    out["_session_breakdown_str"] = format_session_breakdown(sm)
    return out


def main(epochs: int, seed: int) -> None:
    dataset: Channel6Dataset = load_channel6_dataset()
    full_ds = CsiWindowDataset(dataset.window_index, "identity_or_nonauth", calibration=dataset.calibration)
    # taskF's mask is all-True (see tasks.py), so full_ds.index is window_index itself, same length AND
    # same row order -- required for train_idx/test_idx (positions into window_index, from splits.py)
    # to be valid positions into full_ds directly, with no re-indexing/remapping step.
    assert len(full_ds) == len(dataset.window_index), (len(full_ds), len(dataset.window_index))
    print(f"taskF classes: {full_ds.classes} ({len(full_ds)} windows, aligned 1:1 with window_index)")

    print(f"\n=== Prototypical open-set: leave-one-unauthorized-person-out "
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

    print(f"\n=== Prototypical cross-day: leave-one-day-out (parallel across {len(FOLD_DEVICES)} device(s)) ===")
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
