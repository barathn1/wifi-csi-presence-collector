"""Does deciding from a longer stretch of data (multiple consecutive ~1s windows averaged into one
score) beat deciding from a single ~1s window? Trains crossattn_transformer once (same fold as
run_taskD_focus.py), scores every test window individually, then re-scores after averaging into ~5s
(10 windows) and ~10s (20 windows) segments -- see evaluation/segment_aggregation.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import assert_no_group_leakage
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.evaluation.segment_aggregation import aggregate_scores_by_session
from ml.models.transformer_crossattn import CrossAttentionTransformer
from ml.training.run_taskD_focus import false_accept_breakdown, pick_valid_fold
from ml.training.train import train_classifier

TASK = "taskD_auth_vs_nonauth"


def evaluate_segments(segments: pd.DataFrame) -> dict:
    y_true = (segments["label"] == "authorized").astype(int).values
    scores = segments["aggregated_score"].values
    pred = (scores >= 0.5).astype(int)

    eer, _ = compute_eer(y_true, scores)
    auroc = compute_auroc(y_true, scores)
    accuracy = (pred == y_true).mean()

    breakdown = {}
    for neg_label in ("unauthorized", "none"):
        neg_mask = segments["label"].values == neg_label
        if neg_mask.sum() > 0:
            breakdown[neg_label] = {"n": int(neg_mask.sum()), "false_accept_rate": float(pred[neg_mask].mean())}
    return {"accuracy": float(accuracy), "eer": eer, "auroc": auroc, "breakdown": breakdown, "n_segments": len(segments)}


def print_result(label: str, metrics: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"  n={metrics.get('n_segments', metrics.get('breakdown', {}))}  "
          f"accuracy={metrics['accuracy']:.4f}  EER={metrics['eer']:.4f}  AUROC={metrics['auroc']:.4f}")
    for neg_label, stats in metrics["breakdown"].items():
        print(f"  false-accept as authorized | {neg_label:13s} n={stats['n']:5d} rate={stats['false_accept_rate']:.4f}")


def main(epochs: int = 4) -> None:
    index_path = REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"
    window_index = pd.read_csv(index_path)
    full_ds = CsiWindowDataset(window_index, TASK)

    train_idx, test_idx = pick_valid_fold(full_ds)
    assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
    train_ds = full_ds.subset_by_index_rows(train_idx)
    test_ds = full_ds.subset_by_index_rows(test_idx)
    print(f"fold: n_train={len(train_ds)} n_test={len(test_ds)}")

    model = CrossAttentionTransformer(186, 2)
    train_classifier(model, train_ds, test_ds, epochs=epochs)

    per_window = false_accept_breakdown(model, test_ds)
    print_result("per-window (~1s, single window, no aggregation)", per_window)

    for n_windows, approx_seconds in [(10, "~5s"), (20, "~10s")]:
        segments = aggregate_scores_by_session(test_ds.index, per_window["scores"], n_windows=n_windows)
        metrics = evaluate_segments(segments)
        avg_span = segments["span_packets"].mean() if len(segments) else float("nan")
        print_result(f"segment-aggregated, {n_windows} windows ({approx_seconds}, avg {avg_span:.0f} packets/segment)",
                     metrics)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    args = p.parse_args()
    main(epochs=args.epochs)
