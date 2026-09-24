"""Trains + evaluates the XGBoost baseline (ml_2/models/gbm_baseline.py) on the same channel-6 pooled
dataset and the same 3-tier split discipline as train_svm_gait.py, so it's directly comparable.

    python3 -m ml_2.training.train_gbm
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml_2.data.decode import REPO_ROOT
from ml_2.data.splits import (
    assert_no_group_leakage,
    leave_one_day_out,
    leave_one_unauthorized_person_out,
    session_disjoint_kfold,
)
from ml_2.data.metrics import compute_auroc, compute_eer
from ml_2.models.gbm_baseline import GaitGBM
from ml_2.training.common_data import FEATURE_NAMES, Channel6Dataset, feature_matrix_for, load_channel6_dataset

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/gbm_log.csv"
LOG_FIELDNAMES = ["timestamp", "split_type", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "n_train", "n_test", "notes"]


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def run_fold(X: np.ndarray, rows: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[dict, GaitGBM]:
    train_rows, test_rows = rows.iloc[train_idx], rows.iloc[test_idx]
    y_train = (train_rows["label"] == "authorized").astype(int).values
    y_test = (test_rows["label"] == "authorized").astype(int).values
    model = GaitGBM().fit(X[train_idx], y_train)
    score = model.genuine_score(X[test_idx])

    eer, threshold = compute_eer(y_test, score)
    auroc = compute_auroc(y_test, score)
    pred = (score >= threshold).astype(int) if not np.isnan(threshold) else (score >= 0.5).astype(int)
    out = {"accuracy": float((pred == y_test).mean()), "eer": eer, "auroc": auroc,
           "n_train": len(train_idx), "n_test": len(test_idx)}
    for neg in ("unauthorized", "none"):
        m = test_rows["label"].values == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())

    from ml_2.data.metrics import format_session_breakdown, session_level_metrics
    sm = session_level_metrics(y_test, score, pred, test_rows["session_dir"].values, test_rows["date"].values)
    out["session_accuracy"], out["session_auroc"], out["n_sessions"] = \
        sm["session_accuracy"], sm["session_auroc"], sm["n_sessions"]
    out["_session_breakdown_str"] = format_session_breakdown(sm)
    return out, model


def main() -> None:
    dataset: Channel6Dataset = load_channel6_dataset()
    window_index = dataset.window_index
    X = feature_matrix_for(window_index, dataset)
    print(f"feature matrix: {X.shape}")

    print("\n=== GBM open-set: leave-one-unauthorized-person-out ===")
    fold_metrics = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(window_index, include_none=True):
        assert_no_group_leakage(window_index, train_idx, test_idx)
        m, _ = run_fold(X, window_index, train_idx, test_idx)
        m.update({"split_type": "open_set_loo_stranger", "held_out": held_out})
        fold_metrics.append(m)
        print(f"  held out '{held_out}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f} "
              f"false_accept_unauth={m.get('false_accept_unauthorized', float('nan')):.3f}")
        print(m.pop("_session_breakdown_str", ""))
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": "leave-one-unauthorized-person-out"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")

    print("\n=== GBM cross-day: leave-one-day-out ===")
    fold_metrics = []
    last_model = None
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_index):
        test_labels = window_index.iloc[test_idx]["label"]
        if (test_labels == "authorized").sum() == 0 or len(test_labels.unique()) < 2:
            print(f"  skipping {held_out_date}: degenerate class balance")
            continue
        m, last_model = run_fold(X, window_index, train_idx, test_idx)
        m.update({"split_type": "cross_day_loo", "held_out": held_out_date})
        fold_metrics.append(m)
        print(f"  held out day '{held_out_date}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f}")
        print(m.pop("_session_breakdown_str", ""))
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(), "notes": "leave-one-day-out"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")
    if last_model is not None:
        print("  top-15 features by importance (last fold's model):")
        for name, importance in last_model.top_features(FEATURE_NAMES, k=15):
            print(f"    {name}: {importance:.4f}")

    print("\n=== GBM closed-set sanity: session-disjoint 5-fold ===")
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(session_disjoint_kfold(window_index, n_splits=5)):
        assert_no_group_leakage(window_index, train_idx, test_idx)
        m, _ = run_fold(X, window_index, train_idx, test_idx)
        m.update({"split_type": "session_disjoint_5fold", "held_out": f"fold{fold}"})
        fold_metrics.append(m)
        m.pop("_session_breakdown_str", None)
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(), "notes": "session-disjoint 5-fold"}])
    print(f"  MEAN acc={np.mean([m['accuracy'] for m in fold_metrics]):.3f} "
          f"auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}")

    print(f"\nDone. Results -> {LOG_PATH}")


if __name__ == "__main__":
    main()
