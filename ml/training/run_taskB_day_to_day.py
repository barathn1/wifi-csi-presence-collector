"""Day-to-day identity accuracy for taskB_identity (anjali vs barath), across the model zoo:
RandomForest + SVM (classical, handcrafted per-window features) and CNN, LSTM, BiLSTM, Transformer
(deep, on raw amplitude/phase). Train on Day 1 (2026-09-09), test on Day 2 (2026-09-10) --
`day_disjoint_split` (data_pipeline/splits.py), the actual generalization question, not a
same-day session-disjoint CV number.

Day 1 and Day 2 were captured in DIFFERENT CSI modes (186 subcarriers / HT40 on Day 1, 128 / HT20
on Day 2 -- checked with `decode_csi.py --buckets` on sessions from both days). windowing.py's
existing cache assumes one fixed csi_len (372 bytes) for the whole dataset, which would silently
drop nearly all of Day 2's authorized packets. This script instead caches each session's OWN
dominant bucket and crops to the first N_SUB_COMMON=128 subcarriers (Day 2 sessions are already
128 and pass through unchanged; Day 1 sessions are cropped from 186). This makes shapes match, but
subcarrier index i is not guaranteed to mean the same physical frequency bin across the two modes --
flag any day-to-day accuracy gap as possibly reflecting this capture-mode shift, not purely person
generalization.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from ml.data_pipeline.decode_csi import REPO_ROOT, load_session, decode_session_by_bucket
from ml.data_pipeline.windowing import load_manifest, load_window  # noqa: F401 (load_window used via torch_dataset)
from ml.data_pipeline.splits import day_disjoint_split, assert_no_group_leakage
from ml.data_pipeline.tasks import taskB_identity
from ml.data_pipeline.features import build_feature_matrix
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.models.baselines import make_baseline, make_svm
from ml.models.cnn1d import CNN1DDualBranch
from ml.models.lstm import LSTMDualBranch, BiLSTMDualBranch
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import train_classifier, log_rows

N_SUB_COMMON = 128  # crop target -- see module docstring
CACHE_DIR = REPO_ROOT / "ml/data_pipeline/cache/sessions_taskB_crop128"
WINDOW_PACKETS = 600   # ~3s at the observed 200-250Hz rate, windowing.py's documented "3s variant"
STRIDE_PACKETS = 300   # 50% overlap, project convention (window_packets // 2)
EPOCHS = 6
SEED = 0


def cache_session_cropped(session_dir_rel: str, force: bool = False):
    safe_name = session_dir_rel.replace("/", "__")
    out_path = CACHE_DIR / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    buckets = decode_session_by_bucket(session.npz)
    dom_len = max(buckets, key=lambda k: len(buckets[k]["packet_indices"]))
    bucket = buckets[dom_len]
    packet_idx = bucket["packet_indices"]
    n_sub = bucket["n_subcarriers"]
    if n_sub < N_SUB_COMMON:
        raise ValueError(f"{session_dir_rel}: dominant bucket has {n_sub} subcarriers, "
                          f"less than N_SUB_COMMON={N_SUB_COMMON}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        amplitude=bucket["amplitude"][:, :N_SUB_COMMON],
        phase=bucket["phase"][:, :N_SUB_COMMON],
        rssi=session.npz["rssi"][packet_idx].astype(np.float32),
        device_time_us=session.npz["device_time_us"][packet_idx].astype(np.int64),
    )
    return out_path


def build_taskB_window_index() -> pd.DataFrame:
    manifest = load_manifest()
    auth = manifest[manifest["label"] == "authorized"].reset_index(drop=True)

    rows = []
    for _, row in auth.iterrows():
        cache_path = cache_session_cropped(row["session_dir"])
        with np.load(cache_path, mmap_mode="r") as d:
            n = d["amplitude"].shape[0]
            device_time_us = d["device_time_us"]
            if n < WINDOW_PACKETS:
                print(f"  skipping {row['session_dir']}: only {n} packets < window size {WINDOW_PACKETS}")
                continue
            date = row["session_dir"].split("/")[1]
            for start in range(0, n - WINDOW_PACKETS + 1, STRIDE_PACKETS):
                end = start + WINDOW_PACKETS
                rows.append({
                    "cache_path": str(cache_path), "start": start, "end": end,
                    "session_dir": row["session_dir"], "label": row["label"],
                    "person_id": row["person_id"], "motion": row.get("motion", ""),
                    "date": date, "window_start_time_us": int(device_time_us[start]),
                })
    window_index = pd.DataFrame(rows)
    mask, _ = taskB_identity(window_index)
    assert mask.all(), "build_taskB_window_index should only ever contain authorized rows"
    return window_index


def run_classical(window_index: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                   model_name: str, model_factory, calibration=None) -> dict:
    print(f"  [{model_name}] building handcrafted feature matrix ({len(window_index)} windows)...")
    X = build_feature_matrix(window_index, calibration=calibration)
    y = window_index["person_id"].values
    model = model_factory()
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    y_test = y[test_idx]
    n_correct = int((pred == y_test).sum())
    anjali_mask = y_test == "anjali"
    n_anjali_correct = int((pred[anjali_mask] == y_test[anjali_mask]).sum())
    n_anjali_total = int(anjali_mask.sum())
    return {"model": model_name, "accuracy": n_correct / len(test_idx),
            "anjali_recall": n_anjali_correct / n_anjali_total if n_anjali_total else float("nan"),
            "n_train": len(train_idx), "n_test": len(test_idx),
            "n_correct": n_correct, "n_anjali_correct": n_anjali_correct, "n_anjali_total": n_anjali_total}


def run_deep(window_index: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
             model_name: str, model_factory, n_subcarriers: int = N_SUB_COMMON, calibration=None) -> dict:
    full_ds = CsiWindowDataset(window_index, "taskB_identity", calibration=calibration)
    train_ds = full_ds.subset_by_index_rows(train_idx)
    test_ds = full_ds.subset_by_index_rows(test_idx)

    model = model_factory(n_subcarriers, len(full_ds.classes))
    torch.manual_seed(SEED)
    result = train_classifier(model, train_ds, test_ds, epochs=EPOCHS, seed=SEED)

    # anjali-specific recall
    model.eval()
    anjali_idx = full_ds.class_to_idx["anjali"]
    anjali_correct, anjali_total = 0, 0
    from torch.utils.data import DataLoader
    with torch.no_grad():
        for amp, phase, label in DataLoader(test_ds, batch_size=64):
            mask = label == anjali_idx
            if not mask.any():
                continue
            pred = model(amp, phase).argmax(dim=-1)
            anjali_correct += (pred[mask] == label[mask]).sum().item()
            anjali_total += mask.sum().item()
    anjali_recall = anjali_correct / anjali_total if anjali_total else float("nan")

    return {"model": model_name, "accuracy": result["accuracy"], "anjali_recall": anjali_recall,
            "n_train": result["n_train"], "n_test": result["n_test"],
            "n_correct": result["n_correct"], "n_anjali_correct": anjali_correct, "n_anjali_total": anjali_total}


def main() -> None:
    print("building day-to-day (crop-128) taskB window index...")
    window_index = build_taskB_window_index()
    print(f"total windows: {len(window_index)}")
    print(window_index.groupby(["date", "person_id"]).size())

    day_split = day_disjoint_split(window_index)
    if day_split is None:
        raise RuntimeError("need >1 date for day-disjoint split")
    test_date, train_idx, test_idx = day_split
    assert_no_group_leakage(window_index, train_idx, test_idx, "session_dir")
    train_date = sorted(window_index["date"].unique())[0]
    print(f"train date={train_date} ({len(train_idx)} windows), test date={test_date} ({len(test_idx)} windows)")

    results = []
    results.append(run_classical(window_index, train_idx, test_idx, "random_forest", make_baseline))
    results.append(run_classical(window_index, train_idx, test_idx, "svm", make_svm))
    results.append(run_deep(window_index, train_idx, test_idx, "cnn1d", lambda n, c: CNN1DDualBranch(n, c)))
    results.append(run_deep(window_index, train_idx, test_idx, "lstm", lambda n, c: LSTMDualBranch(n, c)))
    results.append(run_deep(window_index, train_idx, test_idx, "bilstm", lambda n, c: BiLSTMDualBranch(n, c)))
    results.append(run_deep(window_index, train_idx, test_idx, "transformer_whofi", lambda n, c: WhoFiTransformer(n, c)))

    print("\n=== day-to-day (train Day1 -> test Day2) taskB_identity results ===")
    print(f"{'model':<20}{'accuracy':>10}{'anjali_recall':>16}")
    for r in results:
        print(f"{r['model']:<20}{r['accuracy']*100:>9.1f}%{r['anjali_recall']*100:>15.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "taskB_day_to_day",
        "task": "taskB_identity", "preprocessing": "crop128", "model": r["model"],
        "split_type": "day_disjoint", "fold": 0, "accuracy": r["accuracy"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; train={train_date} test={test_date}",
    } for r in results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
