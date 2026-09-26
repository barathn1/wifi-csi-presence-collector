"""Same-day (2026-09-21) open-set feasibility check: anjali/barath (enrolled) vs. promoda/sumanth
(genuinely-unseen strangers -- neither appears in training at all), using the 3-simultaneous-receiver
collection. Answers the actual question asked: does this new data let a BiLSTM (this project's
existing architecture, unchanged) do open-set identification at all, before investing in anything more
elaborate (e.g. true multi-receiver sensor fusion, which would need cross-receiver time
synchronization this collection doesn't provide -- each board only has its OWN `device_time_us`).

Two variants of "use the underlying arch" tried here, both pooling all 3 receivers' windows as
independent extra training/test examples (no cross-receiver fusion, no time-sync needed -- each
receiver's windows are just more data about the same person's gait, same idea as more training
sessions):
1. Embedding + centroid threshold (`run_walk_bilstm_openset.py`'s method): train BiLSTMTriplet on
   anjali/barath only, enroll centroids from those embeddings, threshold on distance.
2. One-class autoencoder (`run_walk_bilstm_autoencoder_openset.py`'s method): train
   BiLSTMAutoencoder on anjali/barath only, threshold on reconstruction error.

Caveats that make this a preliminary look, not a validated result -- stated up front rather than
buried in a footnote:
- promoda and sumanth each have exactly ONE walking session (~2-3 min) -- not enough for genuine
  leave-one-stranger-out or any real diversity within "stranger", just "was this specific unseen
  person accepted or rejected".
- Same-day only: this doesn't (and can't) speak to cross-day generalization the way the 15/16/17-Sept
  LODO result did.
- The 4-second-window train/test split for the *enrolled* identity task here is done by TIME within
  the day (not by day), since there is only one day -- some train/test windows can come from the same
  session (overlapping 50%), a data-leakage risk this script's random split does not fully control
  for on the identity-accuracy number specifically. The open-set FAR/FRR numbers (the actual point of
  this script) are unaffected since promoda/sumanth are 100% held out regardless.

    python -m ml.training.run_sept21_openset
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from ml.data_pipeline.walk_bilstm_pipeline import apply_topk, per_window_zscore, select_topk_variance
from ml.data_pipeline.walk_bilstm_sept21_pipeline import ENROLLED_PEOPLE, build_dataset, build_sept21_manifest
from ml.models.bilstm_autoencoder import BiLSTMAutoencoder
from ml.models.bilstm_triplet import BiLSTMTriplet
from ml.training.losses import triplet_semihard_loss
from ml.training.run_walk_bilstm_autoencoder_openset import per_window_recon_error, train_autoencoder
from ml.training.run_walk_bilstm_loo_day import CLASSES, TRIPLET_WEIGHT, WalkWindowDataset, _accuracy
from ml.training.run_walk_bilstm_openset import embed_all, fit_centroids_and_threshold, nearest_centroid


def prepare(manifest, epochs_split_seed: int = 0):
    windows_all, meta = build_dataset(manifest)
    enrolled_mask = meta["is_enrolled"].values
    stranger_mask = ~enrolled_mask

    enrolled_idx = np.flatnonzero(enrolled_mask)
    train_idx, test_idx = train_test_split(enrolled_idx, test_size=0.25, random_state=epochs_split_seed,
                                            stratify=meta.loc[enrolled_mask, "person_id"].values)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    train_raw = windows_all[train_idx]
    test_raw = windows_all[test_idx]
    stranger_raw = windows_all[stranger_mask]

    idx = select_topk_variance(train_raw, k=30)  # fit on TRAIN (enrolled-train-split) only
    train_reduced = per_window_zscore(apply_topk(train_raw, idx))
    test_reduced = per_window_zscore(apply_topk(test_raw, idx))
    stranger_reduced = per_window_zscore(apply_topk(stranger_raw, idx))

    train_labels = meta.loc[train_idx, "person_id"].map(class_to_idx).values
    test_labels = meta.loc[test_idx, "person_id"].map(class_to_idx).values

    print(f"enrolled train={len(train_idx)} test={len(test_idx)}  strangers(held out)={stranger_mask.sum()} "
          f"({meta.loc[stranger_mask, 'person_id'].value_counts().to_dict()})")
    return train_reduced, train_labels, test_reduced, test_labels, stranger_reduced


def run_embedding_centroid(train_reduced, train_labels, test_reduced, test_labels, stranger_reduced,
                            epochs: int, seed: int, percentile: float = 95.0) -> dict:
    torch.manual_seed(seed)
    model = BiLSTMTriplet(n_features=train_reduced.shape[2], num_persons=len(CLASSES))
    from torch.utils.data import DataLoader
    train_loader = DataLoader(WalkWindowDataset(train_reduced, train_labels, augment=True), batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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

    train_acc = _accuracy(model, DataLoader(WalkWindowDataset(train_reduced, train_labels, augment=False), batch_size=64))
    test_acc = _accuracy(model, DataLoader(WalkWindowDataset(test_reduced, test_labels, augment=False), batch_size=64))

    train_emb = embed_all(model, train_reduced)
    test_emb = embed_all(model, test_reduced)
    stranger_emb = embed_all(model, stranger_reduced)
    centroids, threshold = fit_centroids_and_threshold(train_emb, train_labels, len(CLASSES), percentile)
    _, test_dist = nearest_centroid(test_emb, centroids)
    frr = float((test_dist > threshold).mean())
    _, stranger_dist = nearest_centroid(stranger_emb, centroids)
    far = float((stranger_dist <= threshold).mean()) if len(stranger_dist) else float("nan")
    print(f"  [embedding-centroid] train_acc={train_acc*100:.1f}% test_acc={test_acc*100:.1f}% "
          f"threshold={threshold:.4f} FRR={frr*100:.1f}% FAR={far*100:.1f}%")
    return {"method": "embedding-centroid", "train_acc": train_acc, "test_acc": test_acc, "frr": frr, "far": far}


def run_autoencoder(train_reduced, test_reduced, stranger_reduced, epochs: int, seed: int) -> dict:
    model = train_autoencoder(train_reduced, epochs, seed)
    train_err = per_window_recon_error(model, train_reduced)
    test_err = per_window_recon_error(model, test_reduced)
    stranger_err = per_window_recon_error(model, stranger_reduced)
    threshold = float(np.percentile(train_err, 95))
    frr = float((test_err > threshold).mean())
    far = float((stranger_err <= threshold).mean()) if len(stranger_err) else float("nan")
    print(f"  [autoencoder] train_err_mean={train_err.mean():.5f} test_err_mean={test_err.mean():.5f} "
          f"stranger_err_mean={stranger_err.mean():.5f} threshold={threshold:.5f} FRR={frr*100:.1f}% FAR={far*100:.1f}%")
    return {"method": "autoencoder", "train_err_mean": train_err.mean(), "test_err_mean": test_err.mean(),
            "stranger_err_mean": stranger_err.mean(), "frr": frr, "far": far}


def main(epochs: int, seed: int) -> None:
    manifest = build_sept21_manifest()
    print(f"2026-09-21 walking (session,receiver) rows: {len(manifest)}")
    train_reduced, train_labels, test_reduced, test_labels, stranger_reduced = prepare(manifest, seed)

    print("\n=== SAME-DAY (2026-09-21) OPEN-SET: embedding-centroid ===")
    r1 = run_embedding_centroid(train_reduced, train_labels, test_reduced, test_labels, stranger_reduced, epochs, seed)

    print("\n=== SAME-DAY (2026-09-21) OPEN-SET: one-class autoencoder ===")
    r2 = run_autoencoder(train_reduced, test_reduced, stranger_reduced, epochs, seed)

    print(f"\n=== SUMMARY ===\n{r1}\n{r2}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
