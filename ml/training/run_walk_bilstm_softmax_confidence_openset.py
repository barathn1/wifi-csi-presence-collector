"""Zero-retraining open-set test: reuse the plain 2-class softmax BiLSTM
(`run_walk_bilstm_loo_day.py`, already at ~73% cross-day identity accuracy) and check whether
max-softmax-probability alone separates authorized from unauthorized walkers, instead of argmax.
Hypothesis (common folk wisdom about softmax confidence on OOD input): enrolled people should get a
confident ~0.85-0.95 max-prob, unauthorized people a confused ~0.5-0.7, so thresholding max-prob at
~0.70 might catch unauthorized people without retraining anything.

    python -m ml.training.run_walk_bilstm_softmax_confidence_openset
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.data_pipeline.walk_bilstm_pipeline import (
    TARGET_DATES,
    apply_topk,
    build_dataset,
    build_unauthorized_walk_manifest,
    build_walk_manifest,
    per_window_zscore,
    select_topk_variance,
)
from ml.training.run_walk_bilstm_loo_day import WalkWindowDataset, prepare_fold, train_one_fold


@torch.no_grad()
def max_softmax_probs(model, windows: np.ndarray) -> np.ndarray:
    model.eval()
    ds = WalkWindowDataset(windows, np.zeros(len(windows), dtype=np.int64), augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    out = []
    for x, _ in loader:
        _, logits = model(x)
        out.append(F.softmax(logits, dim=-1).max(dim=-1).values.numpy())
    return np.concatenate(out) if out else np.empty(0, np.float32)


def main(epochs: int, seed: int, threshold: float) -> None:
    auth_manifest = build_walk_manifest()
    unauth_manifest = build_unauthorized_walk_manifest()
    windows_all, auth_meta = build_dataset(auth_manifest)
    unauth_windows_all, unauth_meta = build_dataset(unauth_manifest)

    print(f"\n=== SOFTMAX-CONFIDENCE open-set test (threshold={threshold}, no retraining beyond the "
          f"existing 2-class model) ===")
    fars, frrs = [], []
    for held_out in TARGET_DATES:
        train_reduced, train_labels, test_reduced, test_labels = prepare_fold(windows_all, auth_meta, held_out)
        train_acc, test_acc, model = train_one_fold(train_reduced, train_labels, test_reduced, test_labels, epochs, seed)

        train_mask = (auth_meta["date"] != held_out).values
        idx = select_topk_variance(windows_all[train_mask], k=30)
        unauth_test_mask = (unauth_meta["date"] == held_out).values
        unauth_test_reduced = per_window_zscore(apply_topk(unauth_windows_all[unauth_test_mask], idx))

        auth_probs = max_softmax_probs(model, test_reduced)
        unauth_probs = max_softmax_probs(model, unauth_test_reduced)

        frr = float((auth_probs < threshold).mean())
        far = float((unauth_probs >= threshold).mean()) if len(unauth_probs) else float("nan")
        print(f"  [{held_out}] identity_acc={test_acc*100:.1f}%  "
              f"auth max_prob mean={auth_probs.mean():.3f} (std={auth_probs.std():.3f})  "
              f"unauth max_prob mean={unauth_probs.mean():.3f} (std={unauth_probs.std():.3f})  "
              f"FRR={frr*100:.1f}%  FAR={far*100:.1f}%")
        fars.append(far)
        frrs.append(frr)

    print(f"\nSUMMARY: mean FRR={np.mean(frrs)*100:.1f}%  mean FAR={np.mean(fars)*100:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.70)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed, threshold=args.threshold)
