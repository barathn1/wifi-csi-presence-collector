"""Genuine open-set evaluation: train the walking BiLSTM on anjali/barath PLUS a pool of unauthorized
walkers as an explicit third "unauthorized" class, then test on a held-out day's data where every
unauthorized PERSON appearing in that day's test set has been entirely excluded from training (not
just that day's session -- every session of theirs, on any training day) -- a real leave-one/several-
stranger(s)-out design, combined with this project's existing leave-one-day-out cross-day fold.

Follow-up to `run_walk_bilstm_openset.py`'s finding that a post-hoc embedding-centroid threshold gives
100% FAR (every unauthorized sample accepted) when the model is trained on ONLY the 2 enrolled
identities -- unsurprising, since that model had zero training signal about what "not enrolled" looks
like. This script gives it that signal directly (3-way softmax: anjali/barath/unauthorized) and reads
FAR/FRR straight off the trained decision boundary instead of a separately-fit threshold.

This project's own prior work (`run_ablation_ch6_leave_one_stranger_out.py`, a different model/
preprocessing) already found FAR-unauthorized plateaus at ~64-71% even with strangers in training --
this script checks whether that holds for this walking-only/channel-6/BiLSTM+triplet combination too,
or whether it does better.

    python -m ml.training.run_walk_bilstm_stranger_openset
    python -m ml.training.run_walk_bilstm_stranger_openset --epochs 20 --seeds 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.walk_bilstm_pipeline import (
    TARGET_DATES,
    apply_topk,
    build_dataset,
    build_unauthorized_walk_manifest,
    build_walk_manifest,
    per_window_zscore,
    select_topk_variance,
)
from ml.models.bilstm_triplet import BiLSTMTriplet
from ml.training.losses import triplet_semihard_loss
from ml.training.run_walk_bilstm_loo_day import TRIPLET_WEIGHT, WalkWindowDataset, _accuracy

CLASSES = ["anjali", "barath", "unauthorized"]
UNAUTH_IDX = CLASSES.index("unauthorized")


def prepare_stranger_fold(auth_windows_all, auth_meta, unauth_windows_all, unauth_meta, held_out: str, k: int = 30):
    auth_train_mask = (auth_meta["date"] != held_out).values
    auth_test_mask = (auth_meta["date"] == held_out).values
    auth_train_raw, auth_test_raw = auth_windows_all[auth_train_mask], auth_windows_all[auth_test_mask]
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    auth_train_labels = auth_meta.loc[auth_train_mask, "person_id"].map(class_to_idx).values
    auth_test_labels = auth_meta.loc[auth_test_mask, "person_id"].map(class_to_idx).values

    held_out_people = set(unauth_meta.loc[(unauth_meta["date"] == held_out).values, "person_id"])
    unauth_test_mask = (unauth_meta["date"] == held_out).values
    # exclude every session of a held-out-day stranger, on ANY date -- genuinely unseen identity,
    # not just that day's session.
    unauth_train_mask = (unauth_meta["date"] != held_out).values & (~unauth_meta["person_id"].isin(held_out_people)).values
    unauth_train_raw = unauth_windows_all[unauth_train_mask]
    unauth_test_raw = unauth_windows_all[unauth_test_mask]
    unauth_train_labels = np.full(len(unauth_train_raw), UNAUTH_IDX, dtype=np.int64)
    unauth_test_labels = np.full(len(unauth_test_raw), UNAUTH_IDX, dtype=np.int64)

    train_raw = np.concatenate([auth_train_raw, unauth_train_raw], axis=0)
    train_labels = np.concatenate([auth_train_labels, unauth_train_labels], axis=0)

    idx = select_topk_variance(train_raw, k=k)  # fit on TRAIN (auth+unauth pooled) only
    train_reduced = per_window_zscore(apply_topk(train_raw, idx))
    auth_test_reduced = per_window_zscore(apply_topk(auth_test_raw, idx))
    unauth_test_reduced = per_window_zscore(apply_topk(unauth_test_raw, idx))

    excluded_but_in_training_days = held_out_people & set(unauth_meta.loc[unauth_train_mask, "person_id"])
    print(f"  [{held_out}] train unauth pool: {sorted(set(unauth_meta.loc[unauth_train_mask, 'person_id']))} "
          f"({len(unauth_train_raw)} windows) | held-out strangers: {sorted(held_out_people)} "
          f"({len(unauth_test_raw)} windows) -- none of these appear in the training pool above"
          f"{' (sanity: ' + str(excluded_but_in_training_days) + ' should be empty)' if excluded_but_in_training_days else ''}")

    return train_reduced, train_labels, auth_test_reduced, auth_test_labels, unauth_test_reduced


def train_and_eval_fold(train_reduced, train_labels, auth_test_reduced, auth_test_labels, unauth_test_reduced,
                         epochs: int, seed: int, batch_size: int = 32, lr: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    model = BiLSTMTriplet(n_features=train_reduced.shape[2], num_persons=len(CLASSES))
    train_loader = DataLoader(WalkWindowDataset(train_reduced, train_labels, augment=True),
                               batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(WalkWindowDataset(train_reduced, train_labels, augment=False), batch_size=64)
    auth_test_loader = DataLoader(WalkWindowDataset(auth_test_reduced, auth_test_labels, augment=False), batch_size=64)
    unauth_test_loader = DataLoader(
        WalkWindowDataset(unauth_test_reduced, np.full(len(unauth_test_reduced), UNAUTH_IDX, dtype=np.int64), augment=False),
        batch_size=64)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            embedding, logits = model(x)
            loss = ce_loss(logits, y)
            if len(torch.unique(y)) > 1:
                loss = loss + TRIPLET_WEIGHT * triplet_semihard_loss(embedding, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

    train_acc = _accuracy(model, train_eval_loader)

    model.eval()
    with torch.no_grad():
        auth_preds = torch.cat([model(x)[1].argmax(dim=-1) for x, _ in auth_test_loader]).numpy()
        unauth_preds = torch.cat([model(x)[1].argmax(dim=-1) for x, _ in unauth_test_loader]).numpy() \
            if len(unauth_test_reduced) else np.empty(0, dtype=np.int64)

    identity_acc = float((auth_preds == auth_test_labels).mean())
    frr = float((auth_preds == UNAUTH_IDX).mean())  # authorized wrongly rejected
    far = float((unauth_preds != UNAUTH_IDX).mean()) if len(unauth_preds) else float("nan")  # unauthorized wrongly accepted
    return {"train_acc": train_acc, "identity_acc": identity_acc, "frr": frr, "far": far}


def main(epochs: int, seeds: list[int]) -> None:
    auth_manifest = build_walk_manifest()
    unauth_manifest = build_unauthorized_walk_manifest()
    auth_windows_all, auth_meta = build_dataset(auth_manifest)
    unauth_windows_all, unauth_meta = build_dataset(unauth_manifest)
    print(f"authorized windows: {auth_windows_all.shape}, unauthorized windows: {unauth_windows_all.shape}")

    print("\n=== LEAVE-ONE-DAY-OUT + LEAVE-STRANGER(S)-OUT (3-way: anjali/barath/unauthorized) ===")
    fold_results = []
    for held_out in TARGET_DATES:
        train_reduced, train_labels, auth_test_reduced, auth_test_labels, unauth_test_reduced = \
            prepare_stranger_fold(auth_windows_all, auth_meta, unauth_windows_all, unauth_meta, held_out)
        seed_results = [train_and_eval_fold(train_reduced, train_labels, auth_test_reduced, auth_test_labels,
                                             unauth_test_reduced, epochs, seed) for seed in seeds]
        mean = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
        print(f"    train_acc={mean['train_acc']*100:.1f}%  identity_acc={mean['identity_acc']*100:.1f}%  "
              f"FRR={mean['frr']*100:.1f}%  FAR(unauthorized accepted)={mean['far']*100:.1f}%")
        fold_results.append(mean)

    print(f"\n=== SUMMARY across {len(fold_results)} folds ===")
    for k in ("train_acc", "identity_acc", "frr", "far"):
        print(f"  mean {k} = {np.mean([r[k] for r in fold_results])*100:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()
    main(epochs=args.epochs, seeds=args.seeds)
