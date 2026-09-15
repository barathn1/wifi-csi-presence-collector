"""Train fresh models on ONLY Day3 (2026-09-15), ONLY channel-6 sessions -- deliberately NOT pooled
with Day1/Day2 (different, non-overlapping RF bands, see [[project-day2-cross-channel-root-cause]]) or
even with this same day's channel-11 sessions (same reason -- the router was switched mid-day). Channel
membership is read straight from the recorded per-packet channel_primary field (`decode_csi.py
::channel_width_summary`), not assumed from the clock time. Produces its own checkpoint set
(`*_day3ch6.pt`) so it never clobbers `train_final_model.py`'s Day1+2-pooled checkpoints -- both can
coexist and `ml/inference/live_infer.py --checkpoint-suffix _day3ch6` picks these instead.

Two-stage, same "evaluate honestly, then ship" discipline as the rest of this project:
1. Held-out numbers FIRST: leave-one-unauthorized-person-out for taskD (this Day3-ch6 collection has 4
   real, distinct stranger identities -- divya/harshitha/sumanth/abdul -- so this is a genuine open-set
   check, not the auth-vs-empty-room-only proxy a stranger-free dataset would be limited to), a single
   session-disjoint 80/20 split for task0_presence/taskE. These numbers, not the final checkpoint's
   in-sample accuracy, are the trustworthy ones.
2. Only after that: retrain each task on 100% of the ch6 Day3 data for the actual deployable checkpoint
   -- there's no Day4 yet to hold out for, so the step-1 numbers above stand in as "the real test" this
   time (same role Day3 played for train_final_model.py's Day1+2-pooled checkpoints).

    python3 -m ml.training.train_day3_ch6_model
    python3 -m ml.training.train_day3_ch6_model --epochs 6 --seed 1
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.data_pipeline.calibration import apply_variant_a, compute_day_baseline_from_sessions
from ml.data_pipeline.decode_csi import REPO_ROOT, channel_width_summary, load_session
from ml.data_pipeline.splits import (
    assert_no_group_leakage,
    leave_one_unauthorized_person_out,
    session_disjoint_kfold,
)
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.data_pipeline.windowing import build_window_index, load_manifest
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.inference.checkpoint import save_checkpoint
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import train_classifier

DAY3_DATE = "2026-09-15"
TARGET_CHANNEL = 6
INDEX_PATH = REPO_ROOT / f"ml/data_pipeline/cache/window_index_day3ch{TARGET_CHANNEL}_w200_s100.csv"
CHECKPOINT_DIR = REPO_ROOT / "ml/checkpoints"
LOG_PATH = REPO_ROOT / "ml/evaluation/results/day3_ch6_model_log.csv"
LOG_FIELDNAMES = ["timestamp", "stage", "task", "held_out", "accuracy", "eer", "auroc",
                   "false_accept_unauthorized", "false_accept_none", "n_train", "n_test", "notes"]

WINNING_ARCH_KWARGS = dict(d_model=32, n_heads=4, d_ff=64, num_layers=1, norm_first=False, dropout=0.2)

# task_name -> (checkpoint filename, display_labels for the ALPHABETICALLY-SORTED classes)
TASKS = {
    "taskD_auth_vs_nonauth": ("whofi_taskD_calibA_day3ch6.pt", ["NOT AUTHORIZED", "AUTHORIZED"]),
    "task0_presence": ("whofi_task0_presence_calibA_day3ch6.pt", ["EMPTY", "OCCUPIED"]),
    "taskE_motion_standing_vs_walking": ("whofi_taskE_motion_calibA_day3ch6.pt", ["STANDING", "WALKING"]),
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


def session_channel(session_dir: str) -> int:
    session = load_session(REPO_ROOT / "data" / session_dir)
    return channel_width_summary(session.npz)["channel_primary"]


def build_day3_ch6_manifest() -> pd.DataFrame:
    manifest = load_manifest()
    day3 = manifest[manifest["session_dir"].str.contains(f"/{DAY3_DATE}/", regex=False)].copy()
    day3["channel_primary"] = day3["session_dir"].apply(session_channel)
    ch6 = day3[day3["channel_primary"] == TARGET_CHANNEL].drop(columns=["channel_primary"])
    return ch6.reset_index(drop=True)


def build_or_load_window_index(manifest: pd.DataFrame) -> pd.DataFrame:
    if INDEX_PATH.exists():
        return pd.read_csv(INDEX_PATH)
    index = build_window_index(manifest=manifest, mode="resampled", window_packets=200, stride_packets=100)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(INDEX_PATH, index=False)
    return index


@torch.no_grad()
def neural_logits(model: torch.nn.Module, ds: CsiWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    all_logits, all_y = [], []
    for amp, phase, y in loader:
        logits = model(amp, phase)
        all_logits.append(logits.numpy())
        all_y.append(np.asarray(y))
    return np.concatenate(all_logits), np.concatenate(all_y)


def binary_eval(model: torch.nn.Module, test_ds: CsiWindowDataset) -> dict:
    logits, y_true = neural_logits(model, test_ds)
    proba = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    pred = logits.argmax(axis=-1)
    eer, _ = compute_eer(y_true, proba)
    auroc = compute_auroc(y_true, proba)
    out = {"accuracy": float((pred == y_true).mean()), "eer": eer, "auroc": auroc}
    original_labels = test_ds.index["label"].values
    for neg in ("unauthorized", "none"):
        m = original_labels == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())
    return out


def evaluate_taskD(window_index: pd.DataFrame, calibration, epochs: int, seed: int) -> None:
    full_ds = CsiWindowDataset(window_index, "taskD_auth_vs_nonauth", calibration=calibration)
    print(f"\n=== taskD_auth_vs_nonauth: leave-one-unauthorized-person-out ({len(full_ds)} windows) ===")
    fold_metrics = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(full_ds.index):
        assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
        train_ds = full_ds.subset_by_index_rows(train_idx)
        test_ds = full_ds.subset_by_index_rows(test_idx)
        model = WhoFiTransformer(n_subcarriers=128, n_classes=2, **WINNING_ARCH_KWARGS)
        train_classifier(model, train_ds, test_ds, epochs=epochs, seed=seed)
        metrics = binary_eval(model, test_ds)
        metrics.update({"held_out": held_out, "n_train": len(train_ds), "n_test": len(test_ds)})
        fold_metrics.append(metrics)
        print(f"  held out '{held_out}': acc={metrics['accuracy']:.3f} auroc={metrics['auroc']:.3f} "
              f"eer={metrics['eer']:.3f} false_accept={metrics.get('false_accept_unauthorized', float('nan')):.3f} "
              f"(n_train={metrics['n_train']} n_test={metrics['n_test']})")
        log_rows([{**metrics, "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "eval_loo",
                   "task": "taskD_auth_vs_nonauth", "notes": "leave-one-unauthorized-person-out"}])
    mean_auroc = np.nanmean([m["auroc"] for m in fold_metrics])
    mean_fa = np.nanmean([m.get("false_accept_unauthorized", float("nan")) for m in fold_metrics])
    print(f"  MEAN across {len(fold_metrics)} held-out strangers: auroc={mean_auroc:.3f} "
          f"false_accept_rate={mean_fa:.3f}  <-- the trustworthy number, not the final checkpoint's")


def evaluate_single_split(task_name: str, window_index: pd.DataFrame, calibration, epochs: int, seed: int) -> None:
    full_ds = CsiWindowDataset(window_index, task_name, calibration=calibration)
    train_idx, test_idx = next(session_disjoint_kfold(full_ds.index, n_splits=5, seed=seed))
    assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
    train_ds = full_ds.subset_by_index_rows(train_idx)
    test_ds = full_ds.subset_by_index_rows(test_idx)
    model = WhoFiTransformer(n_subcarriers=128, n_classes=len(full_ds.classes), **WINNING_ARCH_KWARGS)
    train_classifier(model, train_ds, test_ds, epochs=epochs, seed=seed)
    metrics = binary_eval(model, test_ds)
    print(f"\n=== {task_name}: session-disjoint 80/20 split ({len(full_ds)} windows) ===")
    print(f"  acc={metrics['accuracy']:.3f} auroc={metrics['auroc']:.3f} "
          f"(n_train={len(train_ds)} n_test={len(test_ds)})")
    log_rows([{**metrics, "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "eval_split",
               "task": task_name, "held_out": "", "n_train": len(train_ds), "n_test": len(test_ds),
               "notes": "session-disjoint 80/20"}])


def train_final(task_name: str, window_index: pd.DataFrame, calibration, epochs: int, seed: int) -> None:
    filename, display_labels = TASKS[task_name]
    full_ds = CsiWindowDataset(window_index, task_name, calibration=calibration)
    classes = full_ds.classes
    assert len(classes) == len(display_labels), (task_name, classes, display_labels)
    trivial_baseline = max((full_ds.y == c).mean() for c in classes)
    model = WhoFiTransformer(n_subcarriers=128, n_classes=len(classes), **WINNING_ARCH_KWARGS)
    result = train_classifier(model, full_ds, full_ds, epochs=epochs, seed=seed)
    print(f"\n=== {task_name}: FINAL checkpoint, trained on ALL {len(full_ds)} ch{TARGET_CHANNEL}-Day3 windows ===")
    print(f"  in-sample sanity accuracy: {result['accuracy']:.4f} (trivial baseline: {trivial_baseline:.4f})")
    out_path = CHECKPOINT_DIR / filename
    save_checkpoint(
        model, out_path, model_class="WhoFiTransformer",
        arch_kwargs=dict(n_subcarriers=128, n_classes=len(classes), **WINNING_ARCH_KWARGS),
        classes=list(classes), display_labels=display_labels, task_name=task_name,
        preprocessing="calibA", mode="resampled", window_packets=200, stride_packets=100,
        train_dates=[DAY3_DATE], seed=seed, epochs=epochs, train_loss_curve=result["train_loss_curve"],
        notes=f"Day3 ({DAY3_DATE}) channel-{TARGET_CHANNEL}-ONLY, no held-out split -- see this run's "
              f"eval_loo/eval_split log rows for the honest held-out numbers",
    )
    print(f"  saved -> {out_path}")
    log_rows([{"timestamp": datetime.now(timezone.utc).isoformat(), "stage": "final", "task": task_name,
               "held_out": "", "accuracy": result["accuracy"], "n_train": result["n_train"],
               "n_test": result["n_test"],
               "notes": f"in-sample sanity only; trivial_baseline={trivial_baseline:.4f}; checkpoint={filename}"}])


def main(epochs: int, seed: int) -> None:
    manifest = build_day3_ch6_manifest()
    print(f"Day3 channel-{TARGET_CHANNEL} sessions: {len(manifest)} "
          f"({manifest['label'].value_counts().to_dict()})")
    window_index = build_or_load_window_index(manifest)
    print(f"window index: {len(window_index)} windows")

    none_sessions = manifest.loc[manifest["label"] == "none", "session_dir"].tolist()
    baseline = compute_day_baseline_from_sessions(none_sessions, label=f"{DAY3_DATE}_ch{TARGET_CHANNEL}")

    def calibration(amp, phase, row):
        return apply_variant_a(amp, phase, baseline)

    evaluate_taskD(window_index, calibration, epochs, seed)
    evaluate_single_split("task0_presence", window_index, calibration, epochs, seed)
    evaluate_single_split("taskE_motion_standing_vs_walking", window_index, calibration, epochs, seed)

    for task_name in TASKS:
        train_final(task_name, window_index, calibration, epochs, seed)

    print(f"\nDone. Checkpoints in {CHECKPOINT_DIR}/, eval+final log at {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(epochs=args.epochs, seed=args.seed)
