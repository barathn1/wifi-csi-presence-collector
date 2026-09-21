"""Open-set test of the actual gap in what's been tried so far: train the triplet loss WITH
unauthorized people's windows included as extra negatives, each keeping its own real-person identity
label (so two different strangers are still treated as different identities, and the same stranger's
own day-to-day windows are still pulled together like any other identity) -- but the 2-unit softmax
head is trained on, and only ever emits, anjali/barath (never a 3rd "unauthorized" class). This is the
one combination `run_walk_bilstm_openset.py` (triplet on enrolled-only, unauthorized never seen in
training) and `run_walk_bilstm_stranger_openset.py` (3-way softmax with an explicit unauthorized class)
didn't cover: unauthorized-as-unlabeled-negative-only. Both prior attempts gave 100% FAR; this is the
one mechanism left that could plausibly push unauthorized embeddings away from the enrolled centroids,
since it's the only one of the three that ever gives the triplet loss a negative gradient pulling
strangers' embeddings away from whichever enrolled anchors land in the same batch.

Held-out day's stranger identities are excluded from training entirely (every session of theirs, any
date) -- genuinely unseen strangers at test time, same rule `run_walk_bilstm_stranger_openset.py` uses.
Centroids/threshold are fit from TRAIN-day enrolled embeddings only (never touching unauthorized
embeddings, training or test) -- exactly the enrollment-phase procedure described for this approach.

    python -m ml.training.run_walk_bilstm_triplet_negatives_openset
    python -m ml.training.run_walk_bilstm_triplet_negatives_openset --epochs 20 --seed 0 --threshold-percentile 95
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
from ml.training.run_walk_bilstm_loo_day import CLASSES, TRIPLET_WEIGHT, WalkWindowDataset, _accuracy
from ml.training.run_walk_bilstm_openset import embed_all, fit_centroids_and_threshold, nearest_centroid

ENROLLED_N = len(CLASSES)  # 2 (anjali, barath) -- the ONLY classes the softmax head ever predicts


def prepare_fold(auth_windows_all, auth_meta, unauth_windows_all, unauth_meta, held_out: str, k: int = 30):
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    auth_train_mask = (auth_meta["date"] != held_out).values
    auth_test_mask = (auth_meta["date"] == held_out).values
    auth_train_raw, auth_test_raw = auth_windows_all[auth_train_mask], auth_windows_all[auth_test_mask]
    auth_train_labels = auth_meta.loc[auth_train_mask, "person_id"].map(class_to_idx).values
    auth_test_labels = auth_meta.loc[auth_test_mask, "person_id"].map(class_to_idx).values

    held_out_people = set(unauth_meta.loc[(unauth_meta["date"] == held_out).values, "person_id"])
    unauth_train_mask = (unauth_meta["date"] != held_out).values & (~unauth_meta["person_id"].isin(held_out_people)).values
    unauth_test_mask = (unauth_meta["date"] == held_out).values
    unauth_train_raw = unauth_windows_all[unauth_train_mask]
    unauth_test_raw = unauth_windows_all[unauth_test_mask]

    # each real stranger keeps their OWN identity label (>= ENROLLED_N), never a shared "unauthorized"
    # label -- so the triplet loss still treats two different strangers as different identities, it's
    # only the softmax head (never trained on these rows) that has no unit for them at all.
    train_people = sorted(set(unauth_meta.loc[unauth_train_mask, "person_id"]))
    person_to_label = {p: ENROLLED_N + i for i, p in enumerate(train_people)}
    unauth_train_labels = unauth_meta.loc[unauth_train_mask, "person_id"].map(person_to_label).values.astype(np.int64)

    train_raw = np.concatenate([auth_train_raw, unauth_train_raw], axis=0)
    train_labels = np.concatenate([auth_train_labels, unauth_train_labels])
    is_enrolled_row = np.concatenate([np.ones(len(auth_train_raw), bool), np.zeros(len(unauth_train_raw), bool)])

    idx = select_topk_variance(train_raw, k=k)
    train_reduced = per_window_zscore(apply_topk(train_raw, idx))
    auth_test_reduced = per_window_zscore(apply_topk(auth_test_raw, idx))
    unauth_test_reduced = per_window_zscore(apply_topk(unauth_test_raw, idx))

    print(f"  [{held_out}] train: {is_enrolled_row.sum()} enrolled + {len(unauth_train_raw)} unauthorized-negative "
          f"windows ({len(train_people)} distinct strangers: {train_people}) | held-out strangers: {sorted(held_out_people)}")
    return train_reduced, train_labels, is_enrolled_row, auth_test_reduced, auth_test_labels, unauth_test_reduced


class _EnrolledFlaggedDataset(Dataset):
    """Same augmentation semantics as `WalkWindowDataset`, plus an `is_enrolled` flag per row so the
    training loop can mask the softmax cross-entropy to enrolled (anjali/barath) rows only, while
    every row (enrolled or unauthorized-negative) still flows through the triplet loss."""

    def __init__(self, base: WalkWindowDataset, is_enrolled_row: np.ndarray):
        self.base = base
        self.is_enrolled = torch.from_numpy(is_enrolled_row)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, self.is_enrolled[idx]


def train_fold(train_reduced, train_labels, is_enrolled_row, epochs: int, seed: int,
               batch_size: int = 32, lr: float = 1e-3):
    torch.manual_seed(seed)
    model = BiLSTMTriplet(n_features=train_reduced.shape[2], num_persons=ENROLLED_N)
    ds = _EnrolledFlaggedDataset(WalkWindowDataset(train_reduced, train_labels, augment=True), is_enrolled_row)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y, enrolled_mask in loader:
            has_ce = bool(enrolled_mask.any())
            has_triplet = len(torch.unique(y)) > 1
            if not has_ce and not has_triplet:
                continue  # degenerate batch (no enrolled rows and only one identity present)
            opt.zero_grad()
            embedding, logits = model(x)
            loss = ce_loss(logits[enrolled_mask], y[enrolled_mask]) if has_ce else 0.0
            if has_triplet:
                loss = loss + TRIPLET_WEIGHT * triplet_semihard_loss(embedding, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
    return model


def main(epochs: int, seed: int, percentile: float) -> None:
    auth_manifest = build_walk_manifest()
    unauth_manifest = build_unauthorized_walk_manifest()
    auth_windows_all, auth_meta = build_dataset(auth_manifest)
    unauth_windows_all, unauth_meta = build_dataset(unauth_manifest)

    print("\n=== TRIPLET-WITH-UNAUTHORIZED-NEGATIVES open-set test (2-unit softmax, embedding-centroid rejection) ===")
    fars, frrs, id_accs = [], [], []
    for held_out in TARGET_DATES:
        train_reduced, train_labels, is_enrolled_row, auth_test_reduced, auth_test_labels, unauth_test_reduced = \
            prepare_fold(auth_windows_all, auth_meta, unauth_windows_all, unauth_meta, held_out)
        model = train_fold(train_reduced, train_labels, is_enrolled_row, epochs, seed)

        train_eval_loader = DataLoader(WalkWindowDataset(train_reduced[is_enrolled_row], train_labels[is_enrolled_row],
                                                           augment=False), batch_size=64)
        train_acc = _accuracy(model, train_eval_loader)
        test_loader = DataLoader(WalkWindowDataset(auth_test_reduced, auth_test_labels, augment=False), batch_size=64)
        test_acc = _accuracy(model, test_loader)

        train_emb = embed_all(model, train_reduced[is_enrolled_row])
        train_lbls = train_labels[is_enrolled_row]
        test_emb = embed_all(model, auth_test_reduced)
        unauth_emb = embed_all(model, unauth_test_reduced)

        centroids, threshold = fit_centroids_and_threshold(train_emb, train_lbls, ENROLLED_N, percentile)
        test_pred, test_dist = nearest_centroid(test_emb, centroids)
        frr = float((test_dist > threshold).mean())
        _, unauth_dist = nearest_centroid(unauth_emb, centroids)
        far = float((unauth_dist <= threshold).mean()) if len(unauth_dist) else float("nan")

        print(f"    train_acc={train_acc*100:.1f}%  identity_acc(closed-set)={test_acc*100:.1f}%  "
              f"threshold={threshold:.4f}  FRR={frr*100:.1f}%  FAR(unauthorized accepted)={far*100:.1f}%")
        fars.append(far); frrs.append(frr); id_accs.append(test_acc)

    print(f"\nSUMMARY: mean identity_acc={np.mean(id_accs)*100:.1f}%  mean FRR={np.mean(frrs)*100:.1f}%  "
          f"mean FAR={np.mean(fars)*100:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold-percentile", type=float, default=95.0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed, percentile=args.threshold_percentile)
