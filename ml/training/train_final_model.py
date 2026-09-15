"""Train the final, checkpointed models on ALL of Day1+Day2 pooled data -- not a held-out-fold sweep
experiment, a real deployable artifact. Uses the exact winning configuration the Day2 sweep (see
ml/reports/day2_next_steps.md) validated: `whofi_transformer` at its shallow baseline config with
`calibA` (per-day empty-room self-calibration) preprocessing, in `resampled` (128-subcarrier) mode so
Day1 (native 186) and Day2 (native 128, matching the assumed Day 3 bandwidth) are shape-compatible.

No held-out split: the same dataset is passed as both train_ds and test_ds to train_classifier. This is
deliberate, not an oversight -- the user asked for ALL Day1+Day2 data, and Day 3 (tested live via
ml/inference/live_infer.py) is the real test. Carving out even a small internal validation slice would
reintroduce the exact 50%-overlapping-window leakage problem ml/data_pipeline/splits.py exists to
prevent (a naive slice wouldn't be session-disjoint). The printed accuracy is an IN-SAMPLE SANITY CHECK
ONLY -- never a generalization estimate -- compare it against the printed trivial-baseline rate.

    python3 -m ml.training.train_final_model                       # trains all 3 checkpoints
    python3 -m ml.training.train_final_model --task taskD_auth_vs_nonauth --epochs 6
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import pandas as pd

from ml.data_pipeline.calibration import apply_variant_a, compute_day_baseline
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.data_pipeline.windowing import load_manifest
from ml.inference.checkpoint import save_checkpoint
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import train_classifier

INDEX_PATH = REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"  # resampled, both days
CHECKPOINT_DIR = REPO_ROOT / "ml/checkpoints"
LOG_PATH = REPO_ROOT / "ml/evaluation/results/final_model_log.csv"
LOG_FIELDNAMES = ["timestamp", "stage", "task", "preprocessing", "model", "split_type", "fold",
                   "accuracy", "n_train", "n_test", "notes"]


def log_rows(rows: list[dict]) -> None:
    """Own tiny logger (not ml/training/train.py::log_rows, which hardcodes a different path/schema) --
    a small, git-friendly audit trail of retraining history for this specific script, at
    ml/evaluation/results/final_model_log.csv."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})

WINNING_ARCH_KWARGS = dict(d_model=32, n_heads=4, d_ff=64, num_layers=1, norm_first=False, dropout=0.2)

# task_name -> (checkpoint filename, display_labels for the ALPHABETICALLY-SORTED classes)
TASKS = {
    "taskD_auth_vs_nonauth": ("whofi_taskD_calibA.pt", ["NOT AUTHORIZED", "AUTHORIZED"]),
    "task0_presence": ("whofi_task0_presence_calibA.pt", ["EMPTY", "OCCUPIED"]),
    "taskE_motion_standing_vs_walking": ("whofi_taskE_motion_calibA.pt", ["STANDING", "WALKING"]),
    # WARNING (see ml/reports/day2_next_steps.md): this task tested BELOW CHANCE cross-day
    # (AUROC 0.20-0.31, where 0.5 is a coin flip) -- included because it was asked for, but
    # ml/inference/live_infer.py labels its output as experimental/unreliable, not a trustworthy signal.
    "taskB_identity": ("whofi_taskB_identity_calibA.pt", ["ANJALI", "BARATH"]),
}


def calibA_fn(manifest: pd.DataFrame, dates: list[str], mode: str = "resampled"):
    """Mirrors ml/training/run_day2_sweep.py::calibA_fn exactly -- duplicated (not imported) so this
    final-model script doesn't depend on the exploratory sweep script."""
    baselines = {d: compute_day_baseline(manifest, d, mode=mode) for d in dates}

    def fn(amp, phase, row):
        return apply_variant_a(amp, phase, baselines[row["date"]])
    return fn


def train_one(task_name: str, epochs: int, seed: int) -> None:
    filename, display_labels = TASKS[task_name]
    manifest = load_manifest()
    window_index = pd.read_csv(INDEX_PATH)
    dates = sorted(window_index["date"].unique())
    calibration = calibA_fn(manifest, dates, mode="resampled")

    full_ds = CsiWindowDataset(window_index, task_name, calibration=calibration)
    classes = full_ds.classes
    assert len(classes) == len(display_labels), (task_name, classes, display_labels)

    trivial_baseline = max((full_ds.y == c).mean() for c in classes)
    print(f"=== {task_name}: {len(full_ds)} windows, classes={classes}, "
          f"trivial-always-predict-majority baseline={trivial_baseline:.4f} ===")

    model = WhoFiTransformer(n_subcarriers=128, n_classes=len(classes), **WINNING_ARCH_KWARGS)
    result = train_classifier(model, full_ds, full_ds, epochs=epochs, seed=seed)
    print(f"  in-sample sanity accuracy (NOT a generalization estimate): {result['accuracy']:.4f}"
          f"  (trivial baseline: {trivial_baseline:.4f})")

    out_path = CHECKPOINT_DIR / filename
    save_checkpoint(
        model, out_path, model_class="WhoFiTransformer",
        arch_kwargs=dict(n_subcarriers=128, n_classes=len(classes), **WINNING_ARCH_KWARGS),
        classes=list(classes), display_labels=display_labels, task_name=task_name,
        preprocessing="calibA", mode="resampled", window_packets=200, stride_packets=100,
        train_dates=list(dates), seed=seed, epochs=epochs, train_loss_curve=result["train_loss_curve"],
        notes="ALL Day1+Day2 pooled, no held-out split -- Day 3 (tested live) is the real test",
    )
    print(f"  saved -> {out_path}")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "final", "task": task_name,
        "preprocessing": "calibA", "model": "whofi_transformer", "split_type": "none_all_data_pooled",
        "fold": 0, "accuracy": result["accuracy"], "n_train": result["n_train"], "n_test": result["n_test"],
        "notes": f"in-sample sanity only; trivial_baseline={trivial_baseline:.4f}; checkpoint={filename}",
    }])


def main(task: str | None, epochs: int, seed: int) -> None:
    tasks = [task] if task else list(TASKS.keys())
    for t in tasks:
        train_one(t, epochs=epochs, seed=seed)
    print(f"\nDone. Checkpoints in {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=list(TASKS.keys()), default=None,
                   help="train just one task; default trains all 3")
    p.add_argument("--epochs", type=int, default=4,
                   help="matches run_day2_sweep.py's default -- the epoch count behind the validated result")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(task=args.task, epochs=args.epochs, seed=args.seed)
