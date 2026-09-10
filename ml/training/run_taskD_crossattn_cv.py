"""Full session-disjoint CV for crossattn_transformer on taskD_auth_vs_nonauth -- follow-up to
run_taskD_focus.py's single-split result (AUROC 0.757, beating RandomForest's 0.697 on that fold).
That was one split; this checks whether the win holds across every valid fold before trusting it as the
flagship result of the whole sweep. Skips any fold with zero authorized windows in test (same issue
Stage 1 hit -- session_disjoint_kfold isn't label-stratified).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import assert_no_group_leakage, session_disjoint_kfold
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.models.transformer_crossattn import CrossAttentionTransformer
from ml.training.train import log_rows, train_classifier
from ml.training.run_taskD_focus import false_accept_breakdown
from datetime import datetime, timezone

TASK = "taskD_auth_vs_nonauth"


def main(epochs: int = 4, n_splits: int = 5) -> None:
    index_path = REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"
    window_index = pd.read_csv(index_path)
    full_ds = CsiWindowDataset(window_index, TASK)

    rows_out = []
    for i, (train_idx, test_idx) in enumerate(session_disjoint_kfold(full_ds.index, n_splits=n_splits)):
        if len(np.unique(full_ds.y[test_idx])) < 2:
            print(f"fold {i}: skipped (only one class in test)")
            continue
        assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
        train_ds = full_ds.subset_by_index_rows(train_idx)
        test_ds = full_ds.subset_by_index_rows(test_idx)

        model = CrossAttentionTransformer(186, 2)
        train_classifier(model, train_ds, test_ds, epochs=epochs)
        metrics = false_accept_breakdown(model, test_ds)
        unauth = metrics["breakdown"].get("unauthorized", {}).get("false_accept_rate")
        none_far = metrics["breakdown"].get("none", {}).get("false_accept_rate")
        print(f"fold {i}: acc={metrics['accuracy']:.4f} EER={metrics['eer']:.4f} AUROC={metrics['auroc']:.4f} "
              f"false-accept unauth={unauth} none={none_far}")
        rows_out.append({
            "timestamp": datetime.now(timezone.utc).isoformat(), "stage": 2, "task": TASK,
            "preprocessing": "raw", "model": "crossattn_transformer_fullcv",
            "split_type": "session_disjoint_kfold", "fold": i,
            "accuracy": metrics["accuracy"], "eer": metrics["eer"], "auroc": metrics["auroc"],
            "n_train": len(train_idx), "n_test": len(test_idx),
            "notes": f"false_accept_unauth={unauth} false_accept_none={none_far}",
        })

    log_rows(rows_out)
    aucs = [r["auroc"] for r in rows_out]
    print(f"\nmean AUROC={np.mean(aucs):.4f} (n_folds={len(aucs)})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    args = p.parse_args()
    main(epochs=args.epochs)
