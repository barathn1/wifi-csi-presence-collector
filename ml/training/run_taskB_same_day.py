"""Same-day comparison point for run_taskB_day_to_day.py's cross-day (Day1->Day2) collapse: for each
day separately, a single session-disjoint 80/20 split (same convention as train.py's deep-model sweep)
on that day's OWN native CSI mode -- no cropping, since within one day every authorized session shares
the same dominant subcarrier count (186 on Day 1, 128 on Day 2; see run_taskB_day_to_day.py's
docstring). This isolates whether the cross-day collapse is the crop-to-128 artifact / genuine
day-to-day drift, versus these models simply being unable to learn anjali-vs-barath at all.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT, load_session, decode_session_by_bucket
from ml.data_pipeline.windowing import load_manifest
from ml.data_pipeline.splits import session_disjoint_kfold, assert_no_group_leakage
from ml.data_pipeline.tasks import taskB_identity
from ml.models.baselines import make_baseline, make_svm
from ml.models.cnn1d import CNN1DDualBranch
from ml.models.lstm import LSTMDualBranch, BiLSTMDualBranch
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import log_rows
from ml.training.run_taskB_day_to_day import run_classical, run_deep, WINDOW_PACKETS, STRIDE_PACKETS

CACHE_DIR = REPO_ROOT / "ml/data_pipeline/cache/sessions_taskB_native"
N_SPLITS = 5
SEED = 42


def cache_session_native(session_dir_rel: str, force: bool = False):
    safe_name = session_dir_rel.replace("/", "__")
    out_path = CACHE_DIR / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    buckets = decode_session_by_bucket(session.npz)
    dom_len = max(buckets, key=lambda k: len(buckets[k]["packet_indices"]))
    bucket = buckets[dom_len]
    packet_idx = bucket["packet_indices"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        amplitude=bucket["amplitude"],
        phase=bucket["phase"],
        rssi=session.npz["rssi"][packet_idx].astype(np.float32),
        device_time_us=session.npz["device_time_us"][packet_idx].astype(np.int64),
    )
    return out_path


def build_day_window_index(day: str) -> tuple[pd.DataFrame, int]:
    manifest = load_manifest()
    day_sessions = manifest[(manifest["label"] == "authorized")
                             & (manifest["session_dir"].str.contains(f"/{day}/"))].reset_index(drop=True)

    rows = []
    n_sub_ref = None
    for _, row in day_sessions.iterrows():
        cache_path = cache_session_native(row["session_dir"])
        with np.load(cache_path, mmap_mode="r") as d:
            n = d["amplitude"].shape[0]
            n_sub = d["amplitude"].shape[1]
            if n_sub_ref is None:
                n_sub_ref = n_sub
            if n_sub != n_sub_ref:
                print(f"  skipping {row['session_dir']}: n_sub={n_sub} != day's reference {n_sub_ref}")
                continue
            device_time_us = d["device_time_us"]
            if n < WINDOW_PACKETS:
                print(f"  skipping {row['session_dir']}: only {n} packets < window size {WINDOW_PACKETS}")
                continue
            for start in range(0, n - WINDOW_PACKETS + 1, STRIDE_PACKETS):
                end = start + WINDOW_PACKETS
                rows.append({
                    "cache_path": str(cache_path), "start": start, "end": end,
                    "session_dir": row["session_dir"], "label": row["label"],
                    "person_id": row["person_id"], "motion": row.get("motion", ""),
                    "date": day, "window_start_time_us": int(device_time_us[start]),
                })
    window_index = pd.DataFrame(rows)
    mask, _ = taskB_identity(window_index)
    assert mask.all(), "build_day_window_index should only ever contain authorized rows"
    return window_index, n_sub_ref


MODEL_SPECS = [
    ("random_forest", "classical", make_baseline),
    ("svm", "classical", make_svm),
    ("cnn1d", "deep", lambda n, c: CNN1DDualBranch(n, c)),
    ("lstm", "deep", lambda n, c: LSTMDualBranch(n, c)),
    ("bilstm", "deep", lambda n, c: BiLSTMDualBranch(n, c)),
    ("transformer_whofi", "deep", lambda n, c: WhoFiTransformer(n, c)),
]


def run_day(day: str) -> list[dict]:
    """Full session-disjoint 5-fold CV, pooling out-of-fold predictions across folds before scoring --
    NOT a single split. With only 8-9 authorized sessions per day, individual GroupKFold folds can (and
    here, on Day 1, always do) end up single-class in the test set -- e.g. Day 1 fold 0's test set is
    213 windows, 100% barath, which would make "accuracy" meaningless for judging identity separation.
    Pooling every fold's held-out predictions guarantees both classes are represented overall."""
    print(f"\n=== {day}: building same-day (native subcarrier count) taskB window index ===")
    window_index, n_sub = build_day_window_index(day)
    print(f"total windows: {len(window_index)}, n_subcarriers={n_sub}")
    print(window_index.groupby("person_id").size())

    totals = {name: {"n_correct": 0, "n_test": 0, "n_anjali_correct": 0, "n_anjali_total": 0}
              for name, _, _ in MODEL_SPECS}

    for fold, (train_idx, test_idx) in enumerate(session_disjoint_kfold(window_index, n_splits=N_SPLITS, seed=SEED)):
        assert_no_group_leakage(window_index, train_idx, test_idx, "session_dir")
        test_classes = window_index.iloc[test_idx]["person_id"].value_counts().to_dict()
        print(f"  fold {fold}: train={len(train_idx)} test={len(test_idx)} test_classes={test_classes}")
        for name, kind, factory in MODEL_SPECS:
            if kind == "classical":
                r = run_classical(window_index, train_idx, test_idx, name, factory)
            else:
                r = run_deep(window_index, train_idx, test_idx, name, factory, n_sub)
            t = totals[name]
            t["n_correct"] += r["n_correct"]
            t["n_test"] += r["n_test"]
            t["n_anjali_correct"] += r["n_anjali_correct"]
            t["n_anjali_total"] += r["n_anjali_total"]

    results = []
    for name, _, _ in MODEL_SPECS:
        t = totals[name]
        results.append({
            "model": name, "day": day,
            "accuracy": t["n_correct"] / t["n_test"],
            "anjali_recall": t["n_anjali_correct"] / t["n_anjali_total"] if t["n_anjali_total"] else float("nan"),
            "n_train": "", "n_test": t["n_test"],
        })
    return results


def main() -> None:
    all_results = []
    for day in ["2026-09-09", "2026-09-10"]:
        all_results.extend(run_day(day))

    print("\n=== same-day (session-disjoint 5-fold CV, pooled out-of-fold predictions) taskB_identity results ===")
    print(f"{'day':<12}{'model':<20}{'accuracy':>10}{'anjali_recall':>16}")
    for r in all_results:
        print(f"{r['day']:<12}{r['model']:<20}{r['accuracy']*100:>9.1f}%{r['anjali_recall']*100:>15.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "taskB_same_day",
        "task": "taskB_identity", "preprocessing": "native", "model": r["model"],
        "split_type": "session_disjoint_5fold_pooled", "fold": "all", "accuracy": r["accuracy"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; day={r['day']}",
    } for r in all_results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
