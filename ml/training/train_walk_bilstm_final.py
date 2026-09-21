"""Train the DEPLOYABLE walking BiLSTM checkpoint: anjali vs barath, on ALL of 2026-09-15/16/17 pooled
(no held-out day -- this IS the shipped artifact, same "no held-out split" philosophy as
`train_final_model.py`/`train_day3_ch6_model.py`). The cross-day generalization ESTIMATE already comes
from `run_walk_bilstm_loo_day.py`'s Leave-One-Day-Out result (~73% mean test accuracy across 3 folds) --
this script's own printed accuracy is an IN-SAMPLE SANITY CHECK ONLY, not a fresh generalization number.

Saves `ml/checkpoints/walk_bilstm_final.pt`: model weights + the top-30-variance subcarrier indices
(fit on this same pooled 3-day set, via `build_walk_dataset_artifact.build_walk_dataset`) + the
preprocessing constants live inference needs to replicate this exact feature pipeline in real time.

    python -m ml.training.train_walk_bilstm_final
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data_pipeline.build_walk_dataset_artifact import build_walk_dataset
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.walk_bilstm_pipeline import (
    BUTTER_CUTOFF_HZ,
    BUTTER_ORDER,
    FINAL_LEN,
    HAMPEL_N_SIGMAS,
    HAMPEL_WINDOW,
    OVERLAP,
    TARGET_CHANNEL,
    TOP_K_SUBCARRIERS,
    WINDOW_SEC,
)
from ml.models.bilstm_triplet import BiLSTMTriplet
from ml.training.losses import triplet_semihard_loss
from ml.training.run_walk_bilstm_loo_day import CLASSES, TRIPLET_WEIGHT, WalkWindowDataset, _accuracy

CHECKPOINT_PATH = REPO_ROOT / "ml/checkpoints/walk_bilstm_final.pt"


def train(final: np.ndarray, labels: np.ndarray, epochs: int, seed: int, batch_size: int = 32,
          lr: float = 1e-3) -> nn.Module:
    torch.manual_seed(seed)
    model = BiLSTMTriplet(n_features=final.shape[2], num_persons=len(CLASSES))
    loader = DataLoader(WalkWindowDataset(final, labels, augment=True), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad()
            embedding, logits = model(x)
            loss = ce_loss(logits, y)
            if len(torch.unique(y)) > 1:
                loss = loss + TRIPLET_WEIGHT * triplet_semihard_loss(embedding, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
        if epoch == epochs - 1 or epoch % 5 == 0:
            acc = _accuracy(model, DataLoader(WalkWindowDataset(final, labels, augment=False), batch_size=64))
            print(f"  epoch {epoch+1}/{epochs}: in-sample sanity accuracy = {acc*100:.1f}%")
    return model


def main(epochs: int, seed: int) -> None:
    final, meta, idx = build_walk_dataset()
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    labels = meta["person_id"].map(class_to_idx).values
    print(f"training on ALL 3 days pooled: {final.shape} windows "
          f"({meta.groupby(['date', 'person_id']).size().to_dict()})")
    print("NOTE: the accuracy below is an IN-SAMPLE SANITY CHECK ONLY -- it's trained and evaluated on "
          "the same data, so it is NOT a generalization estimate. The real cross-day estimate is "
          "run_walk_bilstm_loo_day.py's ~73% mean Leave-One-Day-Out result. This just confirms training "
          "worked at all (should be well above the 2-class ~50-55% majority-class baseline).")

    model = train(final, labels, epochs, seed)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": final.shape[2],
        "num_persons": len(CLASSES),
        "classes": CLASSES,
        "subcarrier_idx": idx,  # into the original 128 decoded subcarriers -- live inference must apply this exact set
        "preprocessing": {
            "target_channel": TARGET_CHANNEL, "window_sec": WINDOW_SEC, "overlap": OVERLAP,
            "final_len": FINAL_LEN, "top_k_subcarriers": TOP_K_SUBCARRIERS,
            "hampel_window": HAMPEL_WINDOW, "hampel_n_sigmas": HAMPEL_N_SIGMAS,
            "butter_cutoff_hz": BUTTER_CUTOFF_HZ, "butter_order": BUTTER_ORDER,
        },
    }, CHECKPOINT_PATH)
    print(f"saved -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
