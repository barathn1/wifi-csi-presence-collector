"""Same-day (2026-09-21) open-set test using Doppler/frequency-domain features
(`doppler_openset_features.py`) instead of the time-domain top-30-variance amplitude+phase features
every other script here uses. Same train/test/stranger split, same model architectures
(BiLSTMTriplet / BiLSTMAutoencoder, both n_features/seq_len-agnostic so no code changes needed), same
evaluation methodology as `run_sept21_openset.py` -- the ONLY thing that changes is what the model
sees per timestep: gait-band STFT magnitude instead of raw amplitude/phase.

    python -m ml.training.run_sept21_doppler_openset
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ml.data_pipeline.doppler_openset_features import build_doppler_dataset, per_window_zscore_doppler
from ml.data_pipeline.walk_bilstm_sept21_pipeline import build_dataset, build_sept21_manifest
from ml.models.bilstm_autoencoder import BiLSTMAutoencoder
from ml.models.bilstm_triplet import BiLSTMTriplet
from ml.training.run_walk_bilstm_autoencoder_openset import per_window_recon_error, train_autoencoder
from ml.training.run_walk_bilstm_loo_day import CLASSES, WalkWindowDataset, _accuracy
from ml.training.run_walk_bilstm_openset import embed_all, fit_centroids_and_threshold, nearest_centroid


def prepare_doppler(manifest, split_seed: int = 0):
    windows_all, meta = build_dataset(manifest)
    print("computing Doppler profiles for all windows (STFT per subcarrier, may take a minute)...")
    doppler_all = per_window_zscore_doppler(build_doppler_dataset(windows_all))
    print(f"doppler_all shape: {doppler_all.shape}")

    enrolled_mask = meta["is_enrolled"].values
    stranger_mask = ~enrolled_mask
    enrolled_idx = np.flatnonzero(enrolled_mask)
    train_idx, test_idx = train_test_split(enrolled_idx, test_size=0.25, random_state=split_seed,
                                            stratify=meta.loc[enrolled_mask, "person_id"].values)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    train_doppler = doppler_all[train_idx]
    test_doppler = doppler_all[test_idx]
    stranger_doppler = doppler_all[stranger_mask]
    train_labels = meta.loc[train_idx, "person_id"].map(class_to_idx).values
    test_labels = meta.loc[test_idx, "person_id"].map(class_to_idx).values

    print(f"enrolled train={len(train_idx)} test={len(test_idx)}  strangers(held out)={stranger_mask.sum()} "
          f"({meta.loc[stranger_mask, 'person_id'].value_counts().to_dict()})")
    return train_doppler, train_labels, test_doppler, test_labels, stranger_doppler


def run_embedding_centroid(train_d, train_labels, test_d, test_labels, stranger_d, epochs, seed,
                            augment: bool, percentile: float = 95.0) -> dict:
    torch.manual_seed(seed)
    model = BiLSTMTriplet(n_features=train_d.shape[2], num_persons=len(CLASSES))
    loader = DataLoader(WalkWindowDataset(train_d, train_labels, augment=augment), batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            _, logits = model(x)
            loss = ce_loss(logits, y)  # pure CE -- triplet already shown unstable at this dataset size
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

    train_acc = _accuracy(model, DataLoader(WalkWindowDataset(train_d, train_labels, augment=False), batch_size=64))
    test_acc = _accuracy(model, DataLoader(WalkWindowDataset(test_d, test_labels, augment=False), batch_size=64))

    train_emb = embed_all(model, train_d)
    test_emb = embed_all(model, test_d)
    stranger_emb = embed_all(model, stranger_d)
    centroids, threshold = fit_centroids_and_threshold(train_emb, train_labels, len(CLASSES), percentile)
    _, test_dist = nearest_centroid(test_emb, centroids)
    frr = float((test_dist > threshold).mean())
    _, stranger_dist = nearest_centroid(stranger_emb, centroids)
    far = float((stranger_dist <= threshold).mean())
    print(f"  [doppler embedding-centroid] train_acc={train_acc*100:.1f}% test_acc={test_acc*100:.1f}% "
          f"threshold={threshold:.4f} FRR={frr*100:.1f}% FAR={far*100:.1f}%")
    print(f"    test(enrolled) dist: mean={test_dist.mean():.4f} range=[{test_dist.min():.4f},{test_dist.max():.4f}]")
    print(f"    stranger dist:       mean={stranger_dist.mean():.4f} range=[{stranger_dist.min():.4f},{stranger_dist.max():.4f}]")
    return {"train_acc": train_acc, "test_acc": test_acc, "frr": frr, "far": far}


def run_autoencoder(train_d, test_d, stranger_d, epochs, seed) -> dict:
    model = train_autoencoder(train_d, epochs, seed)
    train_err = per_window_recon_error(model, train_d)
    test_err = per_window_recon_error(model, test_d)
    stranger_err = per_window_recon_error(model, stranger_d)
    threshold = float(np.percentile(train_err, 95))
    frr = float((test_err > threshold).mean())
    far = float((stranger_err <= threshold).mean())
    print(f"  [doppler autoencoder] test_err_mean={test_err.mean():.5f} stranger_err_mean={stranger_err.mean():.5f} "
          f"threshold={threshold:.5f} FRR={frr*100:.1f}% FAR={far*100:.1f}%")
    return {"frr": frr, "far": far}


def main(epochs: int, seed: int) -> None:
    manifest = build_sept21_manifest()
    train_d, train_labels, test_d, test_labels, stranger_d = prepare_doppler(manifest, seed)

    print("\n=== DOPPLER embedding-centroid, augment=True ===")
    r1 = run_embedding_centroid(train_d, train_labels, test_d, test_labels, stranger_d, epochs, seed, augment=True)

    print("\n=== DOPPLER one-class autoencoder ===")
    r2 = run_autoencoder(train_d, test_d, stranger_d, epochs, seed)

    print(f"\n=== SUMMARY ===\nembedding-centroid: {r1}\nautoencoder: {r2}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
