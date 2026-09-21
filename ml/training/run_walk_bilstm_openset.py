"""Open-set evaluation: can the walking BiLSTM (`run_walk_bilstm_loo_day.py`) tell an UNAUTHORIZED
person apart from the 2 enrolled identities (anjali/barath), not just classify between the two it was
trained on?

The trained classifier's softmax head cannot do this by construction -- it is a closed 2-way softmax,
so it will always force a prediction onto anjali or barath, never "neither" (softmax confidence on
out-of-distribution input is not a reliable rejection signal). This script instead uses the 32-d
embedding the triplet loss was actually trained for: build one CENTROID per enrolled identity from
that identity's TRAINING-day embeddings, derive a rejection THRESHOLD from the TRAINING days' own
genuine (own-centroid) distances only, then apply both, frozen, to the held-out day's:
- authorized windows (same identities, unseen day) -> should mostly be ACCEPTED and correctly matched
  (false rejects = FRR)
- unauthorized windows (divya/harshitha/sumanth/abdul/siva/manas/kishore -- never seen in training,
  same 3 dates/channel 6/walking) -> should mostly be REJECTED (false accepts = FAR, the number this
  script exists to measure)

Same Leave-One-Day-Out structure and per-fold train-only subcarrier selection/normalization as
`run_walk_bilstm_loo_day.py` (reuses its `prepare_fold`/`train_one_fold` so the model and feature
pipeline are identical, not a separate reimplementation) -- the threshold and centroids are yet another
statistic computed from TRAIN days only, same no-leakage rule.

    python -m ml.training.run_walk_bilstm_openset
    python -m ml.training.run_walk_bilstm_openset --epochs 20 --seed 0 --threshold-percentile 95
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
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
from ml.training.run_walk_bilstm_loo_day import CLASSES, WalkWindowDataset, prepare_fold, train_one_fold


@torch.no_grad()
def embed_all(model, windows: np.ndarray) -> np.ndarray:
    """L2-normalized embeddings for every window -- same normalization space
    `triplet_semihard_loss` trains in, so centroid/threshold distances are meaningful."""
    model.eval()
    ds = WalkWindowDataset(windows, np.zeros(len(windows), dtype=np.int64), augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    out = []
    for x, _ in loader:
        embedding, _ = model(x)
        out.append(torch.nn.functional.normalize(embedding, p=2, dim=1).numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, model.embedding.out_features), np.float32)


def fit_centroids_and_threshold(train_embeddings: np.ndarray, train_labels: np.ndarray, n_classes: int,
                                 percentile: float) -> tuple[np.ndarray, float]:
    """Centroids: per-class mean embedding over TRAINING days only. Threshold: the `percentile`-th
    percentile of TRAINING samples' distance to their OWN class centroid (genuine distances) -- e.g.
    the 95th percentile means we accept losing at most ~5% of genuine training matches to keep the
    threshold tight, before it's ever applied to the held-out day."""
    centroids = np.stack([train_embeddings[train_labels == c].mean(axis=0) for c in range(n_classes)])
    own_dist = np.linalg.norm(train_embeddings - centroids[train_labels], axis=1)
    threshold = float(np.percentile(own_dist, percentile))
    return centroids, threshold


def nearest_centroid(embeddings: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dists = np.linalg.norm(embeddings[:, None, :] - centroids[None, :, :], axis=2)  # (N, n_classes)
    nearest_idx = dists.argmin(axis=1)
    nearest_dist = dists.min(axis=1)
    return nearest_idx, nearest_dist


def evaluate_fold(windows_all: np.ndarray, auth_meta, unauth_windows_all: np.ndarray, unauth_meta,
                   held_out: str, epochs: int, seed: int, percentile: float) -> dict:
    train_reduced, train_labels, test_reduced, test_labels = prepare_fold(windows_all, auth_meta, held_out)
    train_acc, test_acc, model = train_one_fold(train_reduced, train_labels, test_reduced, test_labels,
                                                  epochs, seed)

    train_dates = [d for d in TARGET_DATES if d != held_out]
    unauth_test_mask = (unauth_meta["date"] == held_out).values
    unauth_test_raw = unauth_windows_all[unauth_test_mask]
    if len(unauth_test_raw) == 0:
        print(f"  [{held_out}] no unauthorized walking windows on this held-out day -- skipping FAR")
        return {}

    # Frozen train-only subcarrier idx + per-window normalization, re-derived exactly as in
    # `prepare_fold` (train_reduced/test_reduced already have it applied; unauthorized windows need
    # the SAME idx, which prepare_fold doesn't expose -- recompute it here identically).
    train_mask = (auth_meta["date"] != held_out).values
    idx = select_topk_variance(windows_all[train_mask], k=30)
    unauth_test_reduced = per_window_zscore(apply_topk(unauth_test_raw, idx))

    train_emb = embed_all(model, train_reduced)
    test_emb = embed_all(model, test_reduced)
    unauth_emb = embed_all(model, unauth_test_reduced)

    centroids, threshold = fit_centroids_and_threshold(train_emb, train_labels, len(CLASSES), percentile)

    test_pred, test_dist = nearest_centroid(test_emb, centroids)
    test_rejected = test_dist > threshold
    frr = float(test_rejected.mean())
    id_acc_accepted = float((test_pred[~test_rejected] == test_labels[~test_rejected]).mean()) if (~test_rejected).any() else float("nan")

    _, unauth_dist = nearest_centroid(unauth_emb, centroids)
    unauth_rejected = unauth_dist > threshold
    far = float((~unauth_rejected).mean())  # false ACCEPT = NOT rejected

    print(f"  train={train_dates} test={held_out}: threshold={threshold:.4f} "
          f"n_test_auth={len(test_labels)} n_test_unauth={len(unauth_test_reduced)}")
    print(f"    identity_acc(closed-set)={test_acc*100:.1f}%  identity_acc(accepted-only)={id_acc_accepted*100:.1f}%  "
          f"FRR={frr*100:.1f}%  FAR(unauthorized accepted)={far*100:.1f}%")
    return {"held_out": held_out, "threshold": threshold, "identity_acc": test_acc, "frr": frr, "far": far,
            "n_test_auth": len(test_labels), "n_test_unauth": len(unauth_test_reduced)}


def main(epochs: int, seed: int, percentile: float) -> None:
    auth_manifest = build_walk_manifest()
    unauth_manifest = build_unauthorized_walk_manifest()
    print(f"authorized walking sessions: {len(auth_manifest)}, unauthorized walking sessions: {len(unauth_manifest)}")
    print(unauth_manifest.groupby(["date", "person_id"]).size())

    windows_all, auth_meta = build_dataset(auth_manifest)
    unauth_windows_all, unauth_meta = build_dataset(unauth_manifest)
    print(f"authorized windows: {windows_all.shape}, unauthorized windows: {unauth_windows_all.shape}")

    print("\n=== OPEN-SET LEAVE-ONE-DAY-OUT (embedding-centroid rejection vs. unauthorized walkers) ===")
    results = [evaluate_fold(windows_all, auth_meta, unauth_windows_all, unauth_meta, held_out, epochs, seed, percentile)
               for held_out in TARGET_DATES]
    results = [r for r in results if r]

    if results:
        mean_far = np.mean([r["far"] for r in results])
        mean_frr = np.mean([r["frr"] for r in results])
        print(f"\n=== SUMMARY: mean FAR(unauthorized wrongly accepted)={mean_far*100:.1f}%  "
              f"mean FRR(authorized wrongly rejected)={mean_frr*100:.1f}% across {len(results)} folds ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold-percentile", type=float, default=95.0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed, percentile=args.threshold_percentile)
