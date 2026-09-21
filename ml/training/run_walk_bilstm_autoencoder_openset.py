"""One-class anomaly-detection test: train `BiLSTMAutoencoder` on ONLY authorized walking windows
(no unauthorized data, no labels), then threshold on RECONSTRUCTION ERROR to separate authorized from
unauthorized -- bypassing every classifier/embedding-distance approach already tried (all gave 100%
FAR; see `run_walk_bilstm_openset.py`/`run_walk_bilstm_stranger_openset.py`/
`run_walk_bilstm_triplet_negatives_openset.py`). Same Leave-One-Day-Out structure and per-fold
train-only top-30-variance subcarrier selection as those scripts, for a fair comparison.

If reconstruction error still fully overlaps between authorized and unauthorized here, that's strong
evidence the ESP32-S3 channel-6 CSI data itself doesn't carry a separable open-set biometric signal at
this feature resolution -- not a fixable modeling choice.

    python -m ml.training.run_walk_bilstm_autoencoder_openset
    python -m ml.training.run_walk_bilstm_autoencoder_openset --epochs 20 --seed 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.data_pipeline.walk_bilstm_pipeline import (
    TARGET_DATES,
    apply_topk,
    build_dataset,
    build_unauthorized_walk_manifest,
    build_walk_manifest,
    per_window_zscore,
    select_topk_variance,
)
from ml.models.bilstm_autoencoder import BiLSTMAutoencoder


def per_window_recon_error(model: BiLSTMAutoencoder, windows: np.ndarray, batch_size: int = 64) -> np.ndarray:
    model.eval()
    x_all = torch.from_numpy(windows.astype(np.float32))
    loader = DataLoader(TensorDataset(x_all), batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for (x,) in loader:
            recon, _ = model(x)
            mse = ((recon - x) ** 2).mean(dim=(1, 2))  # per-window MSE
            errors.append(mse.numpy())
    return np.concatenate(errors)


def train_autoencoder(train_windows: np.ndarray, epochs: int, seed: int, batch_size: int = 32,
                       lr: float = 1e-3) -> BiLSTMAutoencoder:
    torch.manual_seed(seed)
    model = BiLSTMAutoencoder(n_features=train_windows.shape[2], seq_len=train_windows.shape[1])
    x_all = torch.from_numpy(train_windows.astype(np.float32))
    loader = DataLoader(TensorDataset(x_all), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for (x,) in loader:
            opt.zero_grad()
            recon, _ = model(x)
            loss = mse_loss(recon, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
    return model


def main(epochs: int, seed: int) -> None:
    auth_manifest = build_walk_manifest()
    unauth_manifest = build_unauthorized_walk_manifest()
    auth_windows_all, auth_meta = build_dataset(auth_manifest)
    unauth_windows_all, unauth_meta = build_dataset(unauth_manifest)
    print(f"authorized windows: {auth_windows_all.shape}, unauthorized windows: {unauth_windows_all.shape}")

    print("\n=== ONE-CLASS AUTOENCODER open-set test (reconstruction error, no unauthorized in training) ===")
    all_auth_test_errors, all_unauth_test_errors = [], []
    for held_out in TARGET_DATES:
        train_mask = (auth_meta["date"] != held_out).values
        test_mask = (auth_meta["date"] == held_out).values
        train_raw, test_raw = auth_windows_all[train_mask], auth_windows_all[test_mask]

        idx = select_topk_variance(train_raw, k=30)  # fit on TRAIN days only, same rule as other scripts
        train_reduced = per_window_zscore(apply_topk(train_raw, idx))
        test_reduced = per_window_zscore(apply_topk(test_raw, idx))
        unauth_test_mask = (unauth_meta["date"] == held_out).values
        unauth_test_reduced = per_window_zscore(apply_topk(unauth_windows_all[unauth_test_mask], idx))

        model = train_autoencoder(train_reduced, epochs, seed)
        train_err = per_window_recon_error(model, train_reduced)
        auth_test_err = per_window_recon_error(model, test_reduced)
        unauth_test_err = per_window_recon_error(model, unauth_test_reduced)

        # threshold from TRAIN-only reconstruction error distribution (95th percentile) -- never
        # touching test-day or unauthorized data, same no-leakage rule as every other script here.
        threshold = float(np.percentile(train_err, 95))
        frr = float((auth_test_err > threshold).mean())
        far = float((unauth_test_err <= threshold).mean()) if len(unauth_test_err) else float("nan")

        print(f"  [{held_out}] train_err mean={train_err.mean():.5f}  "
              f"auth_test_err mean={auth_test_err.mean():.5f} (std={auth_test_err.std():.5f})  "
              f"unauth_test_err mean={unauth_test_err.mean():.5f} (std={unauth_test_err.std():.5f})  "
              f"threshold={threshold:.5f}  FRR={frr*100:.1f}%  FAR={far*100:.1f}%")
        all_auth_test_errors.append(auth_test_err)
        all_unauth_test_errors.append(unauth_test_err)

    auth_all = np.concatenate(all_auth_test_errors)
    unauth_all = np.concatenate(all_unauth_test_errors)
    print(f"\nPOOLED (all held-out days): authorized test_err mean={auth_all.mean():.5f} std={auth_all.std():.5f} "
          f"range=[{auth_all.min():.5f},{auth_all.max():.5f}]")
    print(f"POOLED unauthorized test_err mean={unauth_all.mean():.5f} std={unauth_all.std():.5f} "
          f"range=[{unauth_all.min():.5f},{unauth_all.max():.5f}]")
    overlap_lo, overlap_hi = max(auth_all.min(), unauth_all.min()), min(auth_all.max(), unauth_all.max())
    print(f"overlap region: [{overlap_lo:.5f}, {overlap_hi:.5f}]" if overlap_hi > overlap_lo else "NO OVERLAP")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
