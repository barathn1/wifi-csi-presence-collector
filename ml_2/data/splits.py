"""Leakage-safe splits for the (session, receiver, window) index. Grouping is always by `session_dir`
(never `session_dir`+`receiver_mac`) for anything session/person/day-disjoint: a multi-receiver
session's 3 receivers are 3 views of the SAME physical event, so letting one receiver's windows land
in train while another receiver's windows from the exact same walk land in test would leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def session_disjoint_kfold(window_index: pd.DataFrame, n_splits: int = 5):
    groups = window_index["session_dir"].values
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(window_index))
    for train_idx, test_idx in gkf.split(idx, groups=groups):
        yield train_idx, test_idx


def leave_one_unauthorized_person_out(window_index: pd.DataFrame, authorized_test_frac: float = 0.25,
                                       seed: int = 42, include_none: bool = True):
    """Holds out ONE unauthorized person entirely (a stand-in for a genuinely novel stranger), plus a
    fixed session-disjoint slice of authorized and `none` sessions, so every fold's test set has both
    classes (required for EER/AUROC to be defined)."""
    auth_sessions = sorted(window_index.loc[window_index["label"] == "authorized", "session_dir"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(auth_sessions)
    n_test = max(1, int(round(len(auth_sessions) * authorized_test_frac)))
    auth_test_sessions = set(shuffled[:n_test])

    is_auth = window_index["label"] == "authorized"
    auth_test_mask = is_auth & window_index["session_dir"].isin(auth_test_sessions)
    auth_train_mask = is_auth & ~window_index["session_dir"].isin(auth_test_sessions)

    none_test_mask = pd.Series(False, index=window_index.index)
    none_train_mask = pd.Series(False, index=window_index.index)
    if include_none:
        none_sessions = sorted(window_index.loc[window_index["label"] == "none", "session_dir"].unique())
        shuffled_none = rng.permutation(none_sessions)
        n_none_test = max(1, int(round(len(none_sessions) * authorized_test_frac)))
        none_test_sessions = set(shuffled_none[:n_none_test])
        is_none = window_index["label"] == "none"
        none_test_mask = is_none & window_index["session_dir"].isin(none_test_sessions)
        none_train_mask = is_none & ~window_index["session_dir"].isin(none_test_sessions)

    unauth_people = sorted(window_index.loc[window_index["label"] == "unauthorized", "person_id"].unique())
    for held_out in unauth_people:
        is_held_out = (window_index["label"] == "unauthorized") & (window_index["person_id"] == held_out)
        unauth_train_mask = (window_index["label"] == "unauthorized") & ~is_held_out
        test_mask = auth_test_mask | is_held_out | none_test_mask
        train_mask = auth_train_mask | unauth_train_mask | none_train_mask
        yield held_out, np.flatnonzero(train_mask.values), np.flatnonzero(test_mask.values)


def leave_one_day_out(window_index: pd.DataFrame):
    for held_out_date in sorted(window_index["date"].unique()):
        test_mask = (window_index["date"] == held_out_date).values
        yield held_out_date, np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def assert_no_group_leakage(window_index: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    train_sessions = set(window_index.iloc[train_idx]["session_dir"])
    test_sessions = set(window_index.iloc[test_idx]["session_dir"])
    overlap = train_sessions & test_sessions
    assert not overlap, f"leakage: session_dir values in both train and test: {overlap}"
