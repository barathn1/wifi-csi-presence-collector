"""Trains + evaluates the 3 SVM/GMM gait variants (ml_2/models/svm_gait.py) on the channel-6 pooled
dataset, under the two splits that actually matter for this project's real question: open-set (does it
reject a stranger it never trained on) and cross-day (does it hold up on a day it never trained on).
A session-disjoint closed-set number is reported too, as the "easy" sanity check alongside the two hard
ones -- same 3-tier discipline train_day3_ch6_model.py already uses for the deep models.

    python3 -m ml_2.training.train_svm_gait
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
from ml_2.models.svm_gait import BinaryGaitSVM, GaitGmmSvmTwoStage, OneClassGaitSVM
from ml_2.training.common_data import Channel6Dataset, feature_matrix_for, load_channel6_dataset

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/svm_gait_log.csv"
LOG_FIELDNAMES = ["timestamp", "model", "split_type", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "identity_accuracy",
                   "n_train", "n_test", "notes"]


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def _fold_metrics(genuine_score: np.ndarray, y_true: np.ndarray) -> dict:
    eer, threshold = compute_eer(y_true, genuine_score)
    auroc = compute_auroc(y_true, genuine_score)
    pred = (genuine_score >= threshold).astype(int) if not np.isnan(threshold) else (genuine_score >= 0).astype(int)
    out = {"accuracy": float((pred == y_true).mean()), "eer": eer, "auroc": auroc}
    return out, pred


def _false_accepts(pred: np.ndarray, labels: np.ndarray) -> dict:
    out = {}
    for neg in ("unauthorized", "none"):
        m = labels == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())
    return out


MAX_SVM_TRAIN_SAMPLES = 6000  # RBF-kernel SVC/OneClassSVM scale ~O(n^2)-O(n^3); this dataset's folds
# have 100k+ training windows, which would take hours-to-never with sklearn's libsvm backend. Random
# subsampling (seeded) caps fit time to seconds without changing which MODEL is being evaluated.


def _subsample(indices: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=cap, replace=False)


def run_fold(model_name: str, X: np.ndarray, rows: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
             split_type: str, held_out) -> dict:
    train_rows, test_rows = rows.iloc[train_idx], rows.iloc[test_idx]
    y_test = (test_rows["label"] == "authorized").astype(int).values
    X_test = X[test_idx]

    if model_name == "oc_svm":
        auth_idx = _subsample(train_idx[(train_rows["label"] == "authorized").values], MAX_SVM_TRAIN_SAMPLES)
        model = OneClassGaitSVM(nu=0.1).fit(X[auth_idx])
        score = model.genuine_score(X_test)
    elif model_name == "binary_svc":
        sub_idx = _subsample(train_idx, MAX_SVM_TRAIN_SAMPLES)
        sub_rows = rows.iloc[sub_idx]
        model = BinaryGaitSVM().fit(X[sub_idx], (sub_rows["label"] == "authorized").astype(int).values)
        score = model.genuine_score(X_test)
    elif model_name == "gmm_svm_wii":
        auth_idx = _subsample(train_idx[(train_rows["label"] == "authorized").values], MAX_SVM_TRAIN_SAMPLES)
        auth_rows = rows.iloc[auth_idx]
        model = GaitGmmSvmTwoStage().fit(X[auth_idx], X[auth_idx], auth_rows["person_id"].values)
        score = model.genuine_score(X_test)
    else:
        raise ValueError(model_name)

    metrics, pred = _fold_metrics(score, y_test)
    metrics.update(_false_accepts(pred, test_rows["label"].values))

    from ml_2.data.metrics import format_session_breakdown, session_level_metrics
    sm = session_level_metrics(y_test, score, pred, test_rows["session_dir"].values, test_rows["date"].values)
    metrics["session_accuracy"], metrics["session_auroc"], metrics["n_sessions"] = \
        sm["session_accuracy"], sm["session_auroc"], sm["n_sessions"]
    metrics["_session_breakdown_str"] = format_session_breakdown(sm)

    if model_name == "gmm_svm_wii":
        auth_test_mask = test_rows["label"].values == "authorized"
        if auth_test_mask.sum() > 0 and model.has_identity_stage_:
            id_pred = model.predict_identity(X_test[auth_test_mask])
            id_true = test_rows.loc[auth_test_mask, "person_id"].values
            metrics["identity_accuracy"] = float((id_pred == id_true).mean())

    metrics.update({"model": model_name, "split_type": split_type, "held_out": str(held_out),
                     "n_train": len(train_idx), "n_test": len(test_idx)})
    return metrics


def evaluate_open_set(model_name: str, X: np.ndarray, window_index: pd.DataFrame) -> None:
    print(f"\n=== SVM[{model_name}] open-set: leave-one-unauthorized-person-out ===")
    fold_metrics = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(window_index, include_none=True):
        assert_no_group_leakage(window_index, train_idx, test_idx)
        m = run_fold(model_name, X, window_index, train_idx, test_idx, "open_set_loo_stranger", held_out)
        fold_metrics.append(m)
        print(f"  held out '{held_out}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f} "
              f"false_accept_unauth={m.get('false_accept_unauthorized', float('nan')):.3f} "
              f"false_accept_none={m.get('false_accept_none', float('nan')):.3f}")
        print(m.pop("_session_breakdown_str", ""))
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": "leave-one-unauthorized-person-out, includes none"}])
    mean_auroc = np.nanmean([m["auroc"] for m in fold_metrics])
    mean_session_auroc = np.nanmean([m["session_auroc"] for m in fold_metrics])
    mean_fa_u = np.nanmean([m.get("false_accept_unauthorized", np.nan) for m in fold_metrics])
    mean_fa_n = np.nanmean([m.get("false_accept_none", np.nan) for m in fold_metrics])
    print(f"  MEAN across {len(fold_metrics)} held-out strangers: window_auroc={mean_auroc:.3f} "
          f"session_auroc={mean_session_auroc:.3f} "
          f"false_accept_unauth={mean_fa_u:.3f} false_accept_none={mean_fa_n:.3f}")


def evaluate_cross_day(model_name: str, X: np.ndarray, window_index: pd.DataFrame) -> None:
    print(f"\n=== SVM[{model_name}] cross-day: leave-one-day-out ===")
    fold_metrics = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_index):
        test_labels = window_index.iloc[test_idx]["label"]
        if (test_labels == "authorized").sum() == 0 or len(test_labels.unique()) < 2:
            print(f"  skipping {held_out_date}: degenerate class balance this day")
            continue
        m = run_fold(model_name, X, window_index, train_idx, test_idx, "cross_day_loo", held_out_date)
        fold_metrics.append(m)
        print(f"  held out day '{held_out_date}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f} "
              f"false_accept_unauth={m.get('false_accept_unauthorized', float('nan')):.3f} "
              f"false_accept_none={m.get('false_accept_none', float('nan')):.3f}")
        print(m.pop("_session_breakdown_str", ""))
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": "leave-one-day-out (cross-day generalization)"}])
    mean_auroc = np.nanmean([m["auroc"] for m in fold_metrics])
    mean_session_auroc = np.nanmean([m["session_auroc"] for m in fold_metrics])
    print(f"  MEAN across {len(fold_metrics)} held-out days: window_auroc={mean_auroc:.3f} "
          f"session_auroc={mean_session_auroc:.3f}")


def evaluate_closed_set(model_name: str, X: np.ndarray, window_index: pd.DataFrame) -> None:
    print(f"\n=== SVM[{model_name}] closed-set sanity: session-disjoint 5-fold ===")
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(session_disjoint_kfold(window_index, n_splits=5)):
        assert_no_group_leakage(window_index, train_idx, test_idx)
        m = run_fold(model_name, X, window_index, train_idx, test_idx, "session_disjoint_5fold", f"fold{fold}")
        fold_metrics.append(m)
        m.pop("_session_breakdown_str", None)
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "notes": "session-disjoint 5-fold (all folds)"}])
    accs = [m["accuracy"] for m in fold_metrics]
    aurocs = [m["auroc"] for m in fold_metrics]
    print(f"  MEAN across {len(fold_metrics)} folds: acc={np.mean(accs):.3f} auroc={np.nanmean(aurocs):.3f}")


def main() -> None:
    dataset: Channel6Dataset = load_channel6_dataset()
    window_index = dataset.window_index
    print("building handcrafted feature matrix (shared across every classical model in ml_2)...")
    X = feature_matrix_for(window_index, dataset)
    print(f"feature matrix: {X.shape}")

    for model_name in ("oc_svm", "binary_svc", "gmm_svm_wii"):
        evaluate_open_set(model_name, X, window_index)
        evaluate_cross_day(model_name, X, window_index)
        evaluate_closed_set(model_name, X, window_index)

    print(f"\nDone. Results -> {LOG_PATH}")


if __name__ == "__main__":
    main()
