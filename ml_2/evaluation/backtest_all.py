"""Consolidates every ml_2 training script's per-fold log into one comparison table -- "backtest all of
them at once." Run AFTER training/run_all.py (or the individual train_*.py scripts) have produced their
log CSVs under ml_2/evaluation/results/.

    python3 -m ml_2.evaluation.backtest_all
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_2.data.decode import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "ml_2/evaluation/results"
OUT_PATH = RESULTS_DIR / "backtest_summary.csv"

# (log filename, model column if the file has one, else a fixed model name for every row in it)
SOURCES: list[tuple[str, str | None]] = [
    ("svm_gait_log.csv", "model"),
    ("gbm_log.csv", "gbm_xgboost"),
    ("prototypical_log.csv", "prototypical_embedding"),
    ("arcface_log.csv", "arcface_embedding"),
    ("transformer_gpu_log.csv", "model"),  # also covers gnn_subcarrier/radar_cnn, same file
    ("receiver_fusion_gpu_log.csv", "receiver_fusion_3esp"),
    ("cusum_log.csv", "cusum_detector"),
    ("contrastive_log.csv", "contrastive_pretrain_probe"),
    ("maml_log.csv", "maml_reptile"),
    ("generative_hard_negative_log.csv", "model"),
]
METRIC_COLS = ["accuracy", "eer", "auroc", "session_accuracy", "session_auroc",
               "false_accept_unauthorized", "false_accept_none"]


def load_all() -> pd.DataFrame:
    frames = []
    for filename, model_col in SOURCES:
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"  (skipping {filename}: not found yet -- that training script hasn't run)")
            continue
        df = pd.read_csv(path)
        if model_col is None:
            continue
        df["model"] = df[model_col] if model_col in df.columns else model_col
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no ml_2 result logs found under {RESULTS_DIR} -- run training first")
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    for col in METRIC_COLS:
        if col not in df.columns:
            df[col] = np.nan
    grouped = df.groupby(["model", "split_type"])[METRIC_COLS].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    return grouped.reset_index()


def main() -> None:
    df = load_all()
    summary = summarize(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_PATH, index=False)

    print(f"\n{'=' * 100}")
    print("BACKTEST SUMMARY -- every model x every split type, mean across folds")
    print(f"{'=' * 100}")
    labels = {
        "open_set_loo_stranger": "OPEN-SET (leave-one-unauthorized-person-out) -- the real stranger-rejection number",
        "cross_day_loo": "CROSS-DAY (leave-one-day-out) -- the real generalization-to-a-new-day number",
        "cross_day_loo_with_calib": "CROSS-DAY + SHORT CALIBRATION (MAML/Reptile only -- gets a small same-day peek, not a zero-peek number)",
        "session_disjoint_5fold": "CLOSED-SET SANITY (session-disjoint 5-fold) -- the easy number",
    }
    for split_type in labels:
        sub = summary[summary["split_type"] == split_type]
        if sub.empty:
            continue
        label = labels[split_type]
        print(f"\n--- {label} ---")
        for _, row in sub.sort_values("session_auroc_mean", ascending=False).iterrows():
            fa_u = row.get("false_accept_unauthorized_mean", float("nan"))
            fa_n = row.get("false_accept_none_mean", float("nan"))
            print(f"  {row['model']:28s} SESSION_auroc={row.get('session_auroc_mean', float('nan')):.3f}  "
                  f"window_auroc={row['auroc_mean']:.3f} (+/-{row['auroc_std']:.3f})  "
                  f"eer={row['eer_mean']:.3f}  false_accept_unauth={fa_u:.3f}  false_accept_none={fa_n:.3f}  "
                  f"n_folds={int(row['auroc_count'])}")
    print(f"\nFull table -> {OUT_PATH}")


if __name__ == "__main__":
    main()
