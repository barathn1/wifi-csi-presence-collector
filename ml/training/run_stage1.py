"""Stage 1 of the combination sweep: classical RandomForest baseline x all 4 tasks x {raw, Variant-A
calibrated} preprocessing. Cheapest possible path to a first real answer on Day-1 data, and every
number is reported both leakage-safe (session/person-disjoint) and naive-random-split, side by side,
so the inflation gap from a wrong split is visible rather than hidden (RESEARCH_NOTES.md's central
methodological warning, repeated across nearly every cited paper).

For Task B (2 authorized identities), the held-out unit MUST be session, not person: with only 2
identities, a person-disjoint split would put ALL of one person's windows in training and NONE in
test (or vice versa) -- the classifier would never see the class it's asked to predict. Task C (open-set
proxy) is the opposite case: it specifically needs unauthorized PEOPLE held out (to simulate a genuinely
novel intruder) while both authorized identities stay in training every fold -- that's exactly what
`leave_one_unauthorized_person_out` in splits.py does.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import (
    assert_no_group_leakage,
    leave_one_unauthorized_person_out,
    naive_random_split,
    session_disjoint_kfold,
)
from ml.data_pipeline.tasks import TASKS
from ml.models.baselines import evaluate_binary_scores, evaluate_folds
from ml.evaluation.metrics import fold_summary

CACHE_DIR = REPO_ROOT / "ml/data_pipeline/cache"
LOG_PATH = REPO_ROOT / "ml/evaluation/results/experiment_log.csv"
BINARY_TASKS = {"task0_presence", "taskC_openset_proxy", "taskD_auth_vs_nonauth"}


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    fieldnames = ["timestamp", "stage", "task", "preprocessing", "model", "split_type", "fold",
                  "accuracy", "eer", "auroc", "n_train", "n_test", "notes"]
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_task(window_index: pd.DataFrame, X: np.ndarray, task_name: str, preprocessing: str) -> list[dict]:
    mask, y = TASKS[task_name](window_index)
    sub_index = window_index[mask].reset_index(drop=True)
    sub_X = X[mask]
    evaluator = evaluate_binary_scores if task_name in BINARY_TASKS else evaluate_folds

    rows = []
    ts = datetime.now(timezone.utc).isoformat()

    if task_name == "taskC_openset_proxy":
        y_bin = y  # already 0/1 (unauthorized/authorized), aligned with sub_index
        for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(sub_index):
            assert_no_group_leakage(sub_index, train_idx, test_idx, "session_dir")
            metrics = evaluate_binary_scores(sub_X, y_bin, [(train_idx, test_idx)])[0]
            rows.append({"timestamp": ts, "stage": 1, "task": task_name, "preprocessing": preprocessing,
                         "model": "random_forest", "split_type": "leave_one_unauth_person_out",
                         "fold": held_out, **metrics})
    else:
        fold_metrics = []
        for i, (train_idx, test_idx) in enumerate(session_disjoint_kfold(sub_index, n_splits=5)):
            assert_no_group_leakage(sub_index, train_idx, test_idx, "session_dir")
            m = evaluator(sub_X, y, [(train_idx, test_idx)])[0]
            fold_metrics.append(m)
            rows.append({"timestamp": ts, "stage": 1, "task": task_name, "preprocessing": preprocessing,
                         "model": "random_forest", "split_type": "session_disjoint_kfold", "fold": i, **m})
        summary = fold_summary(fold_metrics)
        rows.append({"timestamp": ts, "stage": 1, "task": task_name, "preprocessing": preprocessing,
                     "model": "random_forest", "split_type": "session_disjoint_kfold", "fold": "mean",
                     "accuracy": summary.get("accuracy_mean", ""), "eer": summary.get("eer_mean", ""),
                     "auroc": summary.get("auroc_mean", ""),
                     "notes": f"std={summary.get('accuracy_std', float('nan')):.4f}"})

    # naive random split, for comparison -- NOT the headline number
    train_idx, test_idx = naive_random_split(sub_index)
    m = evaluator(sub_X, y, [(train_idx, test_idx)])[0]
    rows.append({"timestamp": ts, "stage": 1, "task": task_name, "preprocessing": preprocessing,
                 "model": "random_forest", "split_type": "naive_random_LEAKY", "fold": 0, **m,
                 "notes": "leakage-prone, for comparison only"})
    return rows


def main() -> None:
    index_path = CACHE_DIR / "window_index_w200_s100.csv"
    window_index = pd.read_csv(index_path)

    all_rows = []
    for preprocessing, feat_file in [("raw", "features_raw_window_index_w200_s100.npy"),
                                      ("calibA", "features_calibA_window_index_w200_s100.npy")]:
        feat_path = CACHE_DIR / feat_file
        if not feat_path.exists():
            print(f"skipping {preprocessing}: {feat_path} not built yet")
            continue
        X = np.load(feat_path)
        for task_name in TASKS:
            print(f"=== {task_name} | {preprocessing} ===")
            rows = run_task(window_index, X, task_name, preprocessing)
            all_rows.extend(rows)
            for r in rows:
                if r["fold"] in ("mean",) or task_name == "taskC_openset_proxy":
                    print(f"  {r['split_type']:28s} fold={r['fold']:>6} acc={r.get('accuracy'):.4f} "
                          f"eer={r.get('eer', '')} auroc={r.get('auroc', '')}")

    log_rows(all_rows)
    print(f"\nlogged {len(all_rows)} rows -> {LOG_PATH}")


if __name__ == "__main__":
    main()
