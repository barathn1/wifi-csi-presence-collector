"""Trains the self-contained transformer zoo (ml_2/models/transformers.py: WhoFi-style / dual-branch /
cross-attention) on GPU, on the channel-6 dataset where every receiver is its own independent sample,
under the open-set and cross-day splits.

    python3 -m ml_2.training.train_transformer_gpu --epochs 6
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np

from ml_2.data.decode import REPO_ROOT
from ml_2.data.splits import assert_no_group_leakage, leave_one_day_out, leave_one_unauthorized_person_out
from ml_2.data.torch_dataset import CsiWindowDataset
from ml_2.models.transformers import CrossAttentionTransformer, DualBranchTransformer, WhoFiTransformer
from ml_2.training.common_data import Channel6Dataset, N_SUBCARRIERS, load_channel6_dataset
from ml_2.data.metrics import save_session_breakdown_csv
from ml_2.training.gpu_utils import DEVICE, FOLD_DEVICES, binary_eval_gpu, run_folds_parallel, train_classifier_gpu

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/transformer_gpu_log.csv"
SESSION_CSV_PATH = REPO_ROOT / "ml_2/evaluation/results/session_breakdown.csv"
LOG_FIELDNAMES = ["timestamp", "model", "split_type", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "n_train", "n_test", "notes"]
ARCH_KWARGS = dict(d_model=32, n_heads=4, d_ff=64, num_layers=1, norm_first=False, dropout=0.2)
MODEL_FACTORIES = {
    "whofi": lambda n_cls: WhoFiTransformer(n_subcarriers=N_SUBCARRIERS, n_classes=n_cls, **ARCH_KWARGS),
    "dualbranch": lambda n_cls: DualBranchTransformer(n_subcarriers=N_SUBCARRIERS, n_classes=n_cls, **ARCH_KWARGS),
    "crossattn": lambda n_cls: CrossAttentionTransformer(n_subcarriers=N_SUBCARRIERS, n_classes=n_cls, **ARCH_KWARGS),
}


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def _run_one_fold(model_name: str, full_ds: CsiWindowDataset, epochs: int, seed: int,
                   held_out, train_idx, test_idx, device) -> dict:
    train_ds, test_ds = full_ds.subset_by_index_rows(train_idx), full_ds.subset_by_index_rows(test_idx)
    model = MODEL_FACTORIES[model_name](len(full_ds.classes))
    train_classifier_gpu(model, train_ds, test_ds, epochs=epochs, seed=seed, device=device)
    m = binary_eval_gpu(model, test_ds, device=device)
    m.update({"held_out": held_out, "n_train": len(train_ds), "n_test": len(test_ds)})
    return m


def evaluate_open_set(model_name: str, full_ds: CsiWindowDataset, window_index, epochs: int, seed: int) -> None:
    print(f"\n=== {model_name} (GPU) open-set: leave-one-unauthorized-person-out "
          f"(parallel across {len(FOLD_DEVICES)} device(s)) ===")
    fold_specs = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(window_index, include_none=True):
        assert_no_group_leakage(window_index, train_idx, test_idx)
        fold_specs.append((held_out, train_idx, test_idx))

    fold_metrics = run_folds_parallel(
        fold_specs,
        lambda spec, device: _run_one_fold(model_name, full_ds, epochs, seed, *spec, device=device),
    )
    for m in fold_metrics:
        print(f"  held out '{m['held_out']}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f} "
              f"false_accept_unauth={m.get('false_accept_unauthorized', float('nan')):.3f} "
              f"false_accept_none={m.get('false_accept_none', float('nan')):.3f}")
        print(m.pop("_session_breakdown_str", ""))
        per_session_table = m.pop("_per_session_table", None)
        if per_session_table is not None:
            save_session_breakdown_csv(per_session_table, SESSION_CSV_PATH, model_name,
                                        "open_set_loo_stranger", m["held_out"])
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(), "model": model_name,
                    "split_type": "open_set_loo_stranger", "notes": "leave-one-unauthorized-person-out"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")


def evaluate_cross_day(model_name: str, full_ds: CsiWindowDataset, window_index, epochs: int, seed: int) -> None:
    print(f"\n=== {model_name} (GPU) cross-day: leave-one-day-out "
          f"(parallel across {len(FOLD_DEVICES)} device(s)) ===")
    fold_specs = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_index):
        if len(np.unique(full_ds.y[test_idx])) < 2:
            print(f"  skipping {held_out_date}: only one class present")
            continue
        fold_specs.append((held_out_date, train_idx, test_idx))

    fold_metrics = run_folds_parallel(
        fold_specs,
        lambda spec, device: _run_one_fold(model_name, full_ds, epochs, seed, *spec, device=device),
    )
    for m in fold_metrics:
        print(f"  held out day '{m['held_out']}': WINDOW auroc={m['auroc']:.3f} eer={m['eer']:.3f}")
        print(m.pop("_session_breakdown_str", ""))
        per_session_table = m.pop("_per_session_table", None)
        if per_session_table is not None:
            save_session_breakdown_csv(per_session_table, SESSION_CSV_PATH, model_name,
                                        "cross_day_loo", m["held_out"])
        log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(), "model": model_name,
                    "split_type": "cross_day_loo", "notes": "leave-one-day-out"}])
    print(f"  MEAN window_auroc={np.nanmean([m['auroc'] for m in fold_metrics]):.3f}  "
          f"MEAN session_auroc={np.nanmean([m['session_auroc'] for m in fold_metrics]):.3f}")


def main(epochs: int, seed: int, dataset: Channel6Dataset | None = None) -> None:
    print(f"training on device: {DEVICE}")
    dataset = dataset or load_channel6_dataset()
    full_ds = CsiWindowDataset(dataset.window_index, "auth_vs_nonauth", calibration=dataset.calibration)

    for model_name in MODEL_FACTORIES:
        evaluate_open_set(model_name, full_ds, dataset.window_index, epochs, seed)
        evaluate_cross_day(model_name, full_ds, dataset.window_index, epochs, seed)

    print(f"\nDone. Results -> {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
