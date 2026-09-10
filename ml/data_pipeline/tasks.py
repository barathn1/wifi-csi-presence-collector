"""Task label definitions -- map a window_index row to the target for each of the 4 tasks in the plan.

Each function returns (mask, y): `mask` selects which windows apply to the task (e.g. Task B only uses
authorized windows), `y` is the label array for the selected subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def task0_presence(window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """none (empty room) vs occupied (authorized or unauthorized present)."""
    mask = np.ones(len(window_index), dtype=bool)
    y = (window_index["label"] != "none").astype(int).values
    return mask, y


def taskA_threeway(window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """authorized / unauthorized / none, 3-way closed-set classification."""
    mask = np.ones(len(window_index), dtype=bool)
    y = window_index["label"].values
    return mask, y


def taskB_identity(window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """anjali vs barath, among authorized windows only -- the '2 authorized people' question."""
    mask = (window_index["label"] == "authorized").values
    y = window_index.loc[mask, "person_id"].values
    return mask, y


def taskC_openset_proxy(window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """authorized vs unauthorized (binary) -- a closed-set PROXY for open-set verification, meant to be
    evaluated with leave-one-unauthorized-person-out (see splits.py) so at least the "reject a genuinely
    novel identity" axis is tested even before the real embedding+EER model (Stage 2) exists. Per
    RESEARCH_NOTES.md/ARGUS, expect this proxy to overstate open-set performance -- a plain classifier
    structurally cannot say "unknown"; it only shows whether the two classes are separable at all."""
    mask = window_index["label"].isin(["authorized", "unauthorized"]).values
    y = (window_index.loc[mask, "label"] == "authorized").astype(int).values
    return mask, y


def taskD_auth_vs_nonauth(window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """authorized vs EVERYTHING else (unauthorized OR empty room), binary -- the actual production
    framing of this whole project: "is an authorized person here right now, yes/no", collapsing
    unauthorized-person and empty-room into a single negative class rather than treating them as
    separate labels (taskA) or excluding `none` entirely (taskC). This is the single most practically
    relevant number in the sweep."""
    mask = np.ones(len(window_index), dtype=bool)
    y = (window_index["label"] == "authorized").astype(int).values
    return mask, y


TASKS = {
    "task0_presence": task0_presence,
    "taskA_threeway": taskA_threeway,
    "taskB_identity": taskB_identity,
    "taskC_openset_proxy": taskC_openset_proxy,
    "taskD_auth_vs_nonauth": taskD_auth_vs_nonauth,
}
