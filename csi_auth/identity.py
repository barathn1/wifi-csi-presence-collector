"""Anjali vs Barath (closed 2-person identity), on the SAME masked+spike-cleaned dataset/pipeline as
train.py's auth-vs-non-auth task -- reuses train.run()'s leave-one-day-out/model/metrics machinery,
just restricted to the two known people and re-labeled by identity instead of auth/non-auth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataset import build_dataset
from train import run

RESULTS_PATH = "cache/identity_results.csv"


def restrict_to_two_person(window_table: pd.DataFrame, X_stats: np.ndarray, X_seq: np.ndarray):
    mask = window_table["person_id"].isin(["anjali", "barath"]).values
    wt = window_table.loc[mask].reset_index(drop=True)
    y = (wt["person_id"] == "barath").to_numpy().astype(int)
    return wt, X_stats[mask], X_seq[mask], y


if __name__ == "__main__":
    window_table, X_stats, X_seq = build_dataset()
    wt, Xs, Xq, y = restrict_to_two_person(window_table, X_stats, X_seq)
    print(f"anjali vs barath: {len(wt)} windows, {wt['session_dir'].nunique()} sessions, "
          f"{wt['date'].nunique()} days")
    print(wt.groupby(["date", "person_id"])["session_dir"].nunique())

    rows = run(wt, Xs, Xq, y, cnn_epochs=8)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n{len(df)} rows -> {RESULTS_PATH}")
    print(df.groupby("model")[["accuracy", "balanced_accuracy", "f1"]].mean().round(3))
