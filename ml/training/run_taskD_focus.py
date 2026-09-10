"""Focused run on taskD_auth_vs_nonauth (authorized vs everything else) -- the practically central
question of this whole project. Beyond plain accuracy, reports false-accept rate broken down by WHAT
the negative actually was: an empty room vs a real unauthorized person. That split matters enormously
here -- Stage 1's RandomForest run found the aggregate AUROC (~0.79) is propped up almost entirely by
near-perfect empty-room rejection (0-5% false-accept), while real unauthorized people are waved through
as "authorized" 25-44% of the time depending on the fold. A single blended number would hide that.

Uses fold 1 of session_disjoint_kfold, not fold 0 -- fold 0 happens to hold out zero authorized sessions
for this task (session_disjoint_kfold isn't label-stratified), which would make accuracy meaningless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import assert_no_group_leakage, session_disjoint_kfold
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.transformer_crossattn import CrossAttentionTransformer
from ml.models.transformer_dualbranch import DualBranchTransformer
from ml.training.train import DEVICE, train_classifier

TASK = "taskD_auth_vs_nonauth"


def pick_valid_fold(full_ds: CsiWindowDataset, n_splits: int = 5):
    """First fold (in order) where both classes are present in the test split."""
    for train_idx, test_idx in session_disjoint_kfold(full_ds.index, n_splits=n_splits):
        if len(np.unique(full_ds.y[test_idx])) == 2:
            return train_idx, test_idx
    raise RuntimeError("no fold with both classes in test -- increase n_splits or check task balance")


@torch.no_grad()
def false_accept_breakdown(model, test_ds: CsiWindowDataset) -> dict:
    model.eval()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    all_scores, all_pred = [], []
    for amp, phase, _ in loader:
        logits = model(amp.to(DEVICE), phase.to(DEVICE))
        proba = torch.softmax(logits, dim=-1)[:, 1]
        all_scores.append(proba.numpy())
        all_pred.append(logits.argmax(dim=-1).numpy())
    scores = np.concatenate(all_scores)
    pred = np.concatenate(all_pred)

    test_labels = test_ds.index["label"].values
    breakdown = {}
    for neg_label in ("unauthorized", "none"):
        neg_mask = test_labels == neg_label
        if neg_mask.sum() > 0:
            breakdown[neg_label] = {"n": int(neg_mask.sum()), "false_accept_rate": float(pred[neg_mask].mean())}

    y_true = (test_labels == "authorized").astype(int)
    eer, _ = compute_eer(y_true, scores)
    auroc = compute_auroc(y_true, scores)
    accuracy = (pred == y_true).mean()
    return {"accuracy": float(accuracy), "eer": eer, "auroc": auroc, "breakdown": breakdown,
            "scores": scores, "y_true": y_true}


def main(epochs: int = 4) -> None:
    index_path = REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"
    window_index = pd.read_csv(index_path)
    full_ds = CsiWindowDataset(window_index, TASK)

    train_idx, test_idx = pick_valid_fold(full_ds)
    assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
    train_ds = full_ds.subset_by_index_rows(train_idx)
    test_ds = full_ds.subset_by_index_rows(test_idx)
    print(f"fold: n_train={len(train_ds)} n_test={len(test_ds)} "
          f"test label counts: {test_ds.index['label'].value_counts().to_dict()}")

    for name, factory in [("dualbranch_transformer", lambda: DualBranchTransformer(186, 2)),
                           ("crossattn_transformer", lambda: CrossAttentionTransformer(186, 2))]:
        print(f"\n=== {name} ===")
        model = factory()
        result = train_classifier(model, train_ds, test_ds, epochs=epochs)
        metrics = false_accept_breakdown(model, test_ds)
        print(f"  accuracy={metrics['accuracy']:.4f}  EER={metrics['eer']:.4f}  AUROC={metrics['auroc']:.4f}")
        for neg_label, stats in metrics["breakdown"].items():
            print(f"  false-accept as authorized | {neg_label:13s} n={stats['n']:5d} "
                  f"rate={stats['false_accept_rate']:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    args = p.parse_args()
    main(epochs=args.epochs)
