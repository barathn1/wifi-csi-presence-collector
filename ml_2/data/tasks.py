"""Task label definitions -- fresh, minimal set (motion/standing-vs-walking labels aren't reliably
recoverable from metadata.json without leaning on the old pipeline's manifest.csv, so that task is
dropped here rather than guessed at)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def task_presence(window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(window_index), dtype=bool)
    y = (window_index["label"] != "none").astype(int).values
    return mask, y


def task_auth_vs_nonauth(window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """authorized vs EVERYTHING else (unauthorized OR none) -- the production question."""
    mask = np.ones(len(window_index), dtype=bool)
    y = (window_index["label"] == "authorized").astype(int).values
    return mask, y


def task_identity(window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mask = (window_index["label"] == "authorized").values
    y = window_index.loc[mask, "person_id"].values
    return mask, y


def task_identity_or_nonauth(window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """anjali / barath / non_auth -- every row kept, used to train embedding-based open-set models
    (prototypical/arcface) without ever excluding a row (mask is all-True, so row order/positions
    align 1:1 with window_index)."""
    mask = np.ones(len(window_index), dtype=bool)
    is_auth = window_index["label"] == "authorized"
    y = np.where(is_auth, window_index["person_id"].values, "non_auth")
    return mask, y


TASKS = {
    "presence": task_presence,
    "auth_vs_nonauth": task_auth_vs_nonauth,
    "identity": task_identity,
    "identity_or_nonauth": task_identity_or_nonauth,
}
