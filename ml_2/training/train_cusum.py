"""Trains + evaluates the CUSUM adaptive-baseline detector (ml_2/models/cusum_detector.py). Unlike
every other model here, this one is inherently sequential -- it's evaluated by replaying each test
session's windows IN TIME ORDER and letting the detector's baseline adapt as it goes, not by scoring
windows independently/i.i.d.

    python3 -m ml_2.training.train_cusum
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml_2.data.decode import REPO_ROOT
from ml_2.data.metrics import compute_auroc, compute_eer
from ml_2.data.splits import assert_no_group_leakage, leave_one_day_out, leave_one_unauthorized_person_out
from ml_2.models.cusum_detector import CusumDetector
from ml_2.training.common_data import N_SUBCARRIERS, Channel6Dataset, feature_matrix_for, load_channel6_dataset

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/cusum_log.csv"
LOG_FIELDNAMES = ["timestamp", "split_type", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "n_train", "n_test", "notes"]
AMP_MEAN_SLICE = slice(0, N_SUBCARRIERS)  # first N_SUBCARRIERS columns of the feature matrix are per-subcarrier amp_mean


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def run_fold(X: np.ndarray, rows: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    train_rows = rows.iloc[train_idx]
    auth_mask = (train_rows["label"] == "authorized").values
    detector = CusumDetector().fit(X[train_idx][auth_mask][:, AMP_MEAN_SLICE])

    test_rows = rows.iloc[test_idx].copy()
    test_rows["_pos"] = test_idx
    scores = np.empty(len(test_idx))
    for session_dir, group in test_rows.groupby("session_dir"):
        ordered = group.sort_values("start")
        session_scores = detector.score_session(X[ordered["_pos"].values][:, AMP_MEAN_SLICE])
        for pos_in_test, score in zip(np.searchsorted(test_idx, ordered["_pos"].values), session_scores):
            scores[pos_in_test] = score

    y_test = (rows.iloc[test_idx]["label"] == "authorized").astype(int).values
    eer, threshold = compute_eer(y_test, scores)
    auroc = compute_auroc(y_test, scores)
    pred = (scores >= threshold).astype(int) if not np.isnan(threshold) else (scores >= np.median(scores)).astype(int)
    out = {"accuracy": float((pred == y_test).mean()), "eer": eer, "auroc": auroc,
           "n_train": len(train_idx), "n_test": len(test_idx)}
    test_rows_full = rows.iloc[test_idx]
    labels = test_rows_full["label"].values
    for neg in ("unauthorized", "none"):
        m = labels == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())

    from ml_2.data.metrics import format_session_breakdown, session_level_metrics
    sm = session_level_metrics(y_test, scores, pred, test_rows_full["session_dir"].values,
                                test_rows_full["date"].values)
    out["session_accuracy"], out["session_auroc"], out["n_sessions"] = \
        sm["session_accuracy"], sm["session_auroc"], sm["n_sessions"]
    out["_session_breakdown_str"] = format_session_breakdown(sm)
    return out


def main() -> None:
    dataset: Channel6Dataset = load_channel6_dataset()
    window_index = dataset.window_index
    X = feature_matrix_for(window_index, dataset)

    print("\n=== CUSUM open-set: leave-one-unauthorized-person-out ===")
    fold_metrics = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(window_index, include_none=True):
        assert_no_group_leakage(window_index, train_idx, test_idx)
        m = run_fold(X, window_index, train_idx, test_idx)
        m.update({"split_type": "open_set_loo_stranger", "held_out": held_out})
        fold_metrics.append(m)
        print(f"  held out '{held_out}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f}")
        print(m.pop("_session_breakdown_str", ""))
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": "leave-one-unauthorized-person-out, sequential CUSUM replay"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")

    print("\n=== CUSUM cross-day: leave-one-day-out ===")
    fold_metrics = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_index):
        test_labels = window_index.iloc[test_idx]["label"]
        if (test_labels == "authorized").sum() == 0 or len(test_labels.unique()) < 2:
            print(f"  skipping {held_out_date}: degenerate class balance")
            continue
        m = run_fold(X, window_index, train_idx, test_idx)
        m.update({"split_type": "cross_day_loo", "held_out": held_out_date})
        fold_metrics.append(m)
        print(f"  held out day '{held_out_date}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f}")
        print(m.pop("_session_breakdown_str", ""))
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": "leave-one-day-out, sequential CUSUM replay"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")
    print(f"\nDone. Results -> {LOG_PATH}")


if __name__ == "__main__":
    main()
