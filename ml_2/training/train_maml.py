"""Reptile daily-recalibration (ml_2/models/maml_wrapper.py) evaluated the only way that makes sense
for what it actually does: meta-train across every day EXCEPT the held-out one (days-as-tasks), then
adapt using a SHORT calibration slice of the held-out day (simulating "a brief calibration walk each
morning"), then evaluate on the REST of that day. This is a different evaluation contract from every
other model here (which get zero peek at the held-out day) -- disclosed, not hidden: MAML's whole
premise is a small amount of same-day calibration data, so giving it none would test something else.

    python3 -m ml_2.training.train_maml --meta-epochs 10
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_2.data.decode import REPO_ROOT
from ml_2.data.splits import leave_one_day_out
from ml_2.data.torch_dataset import CsiWindowDataset
from ml_2.data.metrics import compute_auroc, compute_eer, save_session_breakdown_csv
from ml_2.models.maml_wrapper import adapt_to_day, reptile_meta_train
from ml_2.models.transformers import WhoFiTransformer
from ml_2.training.common_data import Channel6Dataset, N_SUBCARRIERS, load_channel6_dataset
from ml_2.training.gpu_utils import DEVICE, FOLD_DEVICES, neural_logits_gpu, run_folds_parallel

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/maml_log.csv"
SESSION_CSV_PATH = REPO_ROOT / "ml_2/evaluation/results/session_breakdown.csv"
LOG_FIELDNAMES = ["timestamp", "split_type", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "n_train", "n_calib", "n_test", "notes"]
CALIB_FRACTION = 0.2


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def run_fold(full_ds: CsiWindowDataset, window_index, held_out_date: str, other_train_idx: np.ndarray,
             held_out_idx: np.ndarray, meta_epochs: int, seed: int, device: torch.device = DEVICE) -> dict:
    torch.manual_seed(seed)
    other_dates = sorted(window_index.iloc[other_train_idx]["date"].unique())
    day_tasks = []
    for date in other_dates:
        day_idx = other_train_idx[window_index.iloc[other_train_idx]["date"].values == date]
        loader = DataLoader(full_ds.subset_by_index_rows(day_idx), batch_size=64, shuffle=True, num_workers=4, pin_memory=True, multiprocessing_context="fork")
        day_tasks.append((loader,))

    model = WhoFiTransformer(n_subcarriers=N_SUBCARRIERS, n_classes=len(full_ds.classes),
                              d_model=32, n_heads=4, d_ff=64, num_layers=1, norm_first=False, dropout=0.2)
    model = reptile_meta_train(model, day_tasks, meta_epochs=meta_epochs, device=device)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(held_out_idx)
    n_calib = max(1, int(round(len(shuffled) * CALIB_FRACTION)))
    calib_idx, test_idx = shuffled[:n_calib], shuffled[n_calib:]

    calib_loader = DataLoader(full_ds.subset_by_index_rows(calib_idx), batch_size=32, shuffle=True, num_workers=4, pin_memory=True, multiprocessing_context="fork")
    adapted = adapt_to_day(model, calib_loader, device=device)

    test_ds = full_ds.subset_by_index_rows(test_idx)
    logits, y_idx = neural_logits_gpu(adapted, test_ds, device=device)
    proba = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    pred = logits.argmax(axis=-1)
    y_true = (full_ds.index.iloc[test_idx]["label"].values == "authorized").astype(int)
    eer, _ = compute_eer(y_true, proba)
    auroc = compute_auroc(y_true, proba)
    out = {"accuracy": float((pred == y_true).mean()), "eer": eer, "auroc": auroc,
           "n_train": len(other_train_idx), "n_calib": len(calib_idx), "n_test": len(test_idx)}
    test_rows = full_ds.index.iloc[test_idx]
    labels = test_rows["label"].values
    for neg in ("unauthorized", "none"):
        m = labels == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())

    from ml_2.data.metrics import format_session_breakdown, session_level_metrics
    sm = session_level_metrics(y_true, proba, pred, test_rows["session_dir"].values, test_rows["date"].values)
    out["session_accuracy"], out["session_auroc"], out["n_sessions"] = \
        sm["session_accuracy"], sm["session_auroc"], sm["n_sessions"]
    out["_session_breakdown_str"] = format_session_breakdown(sm)
    out["_per_session_table"] = sm["per_session_table"]
    return out


def main(meta_epochs: int, seed: int) -> None:
    print(f"training on device: {DEVICE}")
    dataset: Channel6Dataset = load_channel6_dataset()
    full_ds = CsiWindowDataset(dataset.window_index, "auth_vs_nonauth", calibration=dataset.calibration)

    print(f"\n=== Reptile daily-recalibration: leave-one-day-out + short same-day calibration "
          f"(parallel across {len(FOLD_DEVICES)} device(s)) ===")
    fold_specs = []
    for held_out_date, other_train_idx, held_out_idx in leave_one_day_out(dataset.window_index):
        held_out_labels = dataset.window_index.iloc[held_out_idx]["label"]
        if (held_out_labels == "authorized").sum() == 0 or len(held_out_labels.unique()) < 2:
            print(f"  skipping {held_out_date}: degenerate class balance")
            continue
        fold_specs.append((held_out_date, other_train_idx, held_out_idx))

    fold_metrics = run_folds_parallel(
        fold_specs,
        lambda spec, device: {**run_fold(full_ds, dataset.window_index, spec[0], spec[1], spec[2],
                                          meta_epochs, seed, device=device), "held_out": spec[0]})
    for m in fold_metrics:
        print(f"  held out day '{m['held_out']}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f} "
              f"(n_calib={m['n_calib']}, n_test={m['n_test']})")
        print(m.pop("_session_breakdown_str", ""))
        per_session_table = m.pop("_per_session_table", None)
        if per_session_table is not None:
            save_session_breakdown_csv(per_session_table, SESSION_CSV_PATH, "maml_reptile",
                                        "cross_day_loo_with_calib", m["held_out"])
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "split_type": "cross_day_loo_with_calib",
                    "notes": "leave-one-day-out, adapted with a 20% same-day calibration slice"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")
    print(f"\nDone. Results -> {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(meta_epochs=args.meta_epochs, seed=args.seed)
