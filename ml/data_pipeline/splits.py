"""Leakage-safe train/test splits for the window index built by windowing.py.

Day 1 has a real confound: every `unauthorized` session was collected back-to-back late in the day
(16:48-17:34), while `authorized`/`none` sessions came earlier (15:05-16:47) -- see RESEARCH_NOTES.md
and the exploration that fed this plan. A plain random split over windows would also leak because
windows overlap 50% by construction (windowing.py's default stride) -- two overlapping windows from the
same session share most of their packets, so they must never land on opposite sides of a split.

This module provides:
- `session_disjoint_kfold` / `person_disjoint_kfold`: the splits actually used for headline numbers.
- `naive_random_split`: the leakage-prone split, kept ONLY so `evaluation/backtest.py` can report it
  side by side with the honest number and make the leakage gap visible (per every paper's warning in
  RESEARCH_NOTES.md about inflated same-session/pooled-random splits).
- `day_disjoint_split`: returns None on Day-1-only data (nothing to hold out yet); once `data/manifest.csv`
  has >1 date, this activates automatically and callers should prefer it over the k-folds above.
- `assert_no_group_leakage`: a hard check, not a comment -- run this after every split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def _person_or_session_group(window_index: pd.DataFrame) -> pd.Series:
    """Group key for person-disjoint splitting: person_id when present, else session_dir (covers
    `none` windows, which have no person_id but must still not straddle a split)."""
    return window_index["person_id"].where(window_index["person_id"] != "", window_index["session_dir"])


def session_disjoint_kfold(window_index: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """GroupKFold on session_dir -- no session's (overlapping) windows appear on both sides."""
    groups = window_index["session_dir"].values
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(window_index))
    for train_idx, test_idx in gkf.split(idx, groups=groups):
        yield train_idx, test_idx


def person_disjoint_kfold(window_index: pd.DataFrame, n_splits: int = 5):
    """GroupKFold on person (falling back to session for `none`) -- for open-set-style tasks where a
    person must never appear in both train and test."""
    groups = _person_or_session_group(window_index).values
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(window_index))
    for train_idx, test_idx in gkf.split(idx, groups=groups):
        yield train_idx, test_idx


def leave_one_unauthorized_person_out(window_index: pd.DataFrame, authorized_test_frac: float = 0.25, seed: int = 42):
    """For open-set verification (Task C): hold out ONE unauthorized person entirely at test time
    (a stand-in for a genuinely novel intruder never seen during training/enrollment/registration).

    The test fold must contain BOTH classes to score EER/AUROC at all -- a fold of only the held-out
    unauthorized person (label always 0) has no positives and every ROC-based metric is undefined on
    it. So a fixed, session-disjoint slice of authorized sessions is held out too (same slice reused
    across every held-out-person fold below): authorized identities stay known/enrolled, but the
    specific *sessions* used to verify them at test time are never the ones trained on, avoiding the
    overlapping-window leakage `session_disjoint_kfold` guards against elsewhere in this module."""
    auth_sessions = sorted(window_index.loc[window_index["label"] == "authorized", "session_dir"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(auth_sessions)
    n_test = max(1, int(round(len(auth_sessions) * authorized_test_frac)))
    auth_test_sessions = set(shuffled[:n_test])

    is_auth = window_index["label"] == "authorized"
    auth_test_mask = is_auth & window_index["session_dir"].isin(auth_test_sessions)
    auth_train_mask = is_auth & ~window_index["session_dir"].isin(auth_test_sessions)

    unauth_people = sorted(window_index.loc[window_index["label"] == "unauthorized", "person_id"].unique())
    for held_out in unauth_people:
        is_held_out_person = (window_index["label"] == "unauthorized") & (window_index["person_id"] == held_out)
        unauth_train_mask = (window_index["label"] == "unauthorized") & ~is_held_out_person

        test_mask = auth_test_mask | is_held_out_person
        train_mask = auth_train_mask | unauth_train_mask
        yield held_out, np.flatnonzero(train_mask.values), np.flatnonzero(test_mask.values)


def naive_random_split(window_index: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """The leakage-prone split (ignores session/person grouping and window overlap) -- report this
    ALONGSIDE the honest split so the inflation gap is visible, never as the only number."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(window_index))
    n_test = int(len(idx) * test_size)
    return idx[n_test:], idx[:n_test]


def day_disjoint_split(window_index: pd.DataFrame):
    """None on Day-1-only data. Once >1 date exists, train on all-but-last date, test on the last."""
    dates = sorted(window_index["date"].unique())
    if len(dates) < 2:
        return None
    test_date = dates[-1]
    train_idx = np.flatnonzero((window_index["date"] != test_date).values)
    test_idx = np.flatnonzero((window_index["date"] == test_date).values)
    return test_date, train_idx, test_idx


def assert_no_group_leakage(window_index: pd.DataFrame, train_idx, test_idx, group_col: str = "session_dir") -> None:
    train_groups = set(window_index.iloc[train_idx][group_col])
    test_groups = set(window_index.iloc[test_idx][group_col])
    overlap = train_groups & test_groups
    assert not overlap, f"leakage: {group_col} values in both train and test: {overlap}"


def time_block_report(window_index: pd.DataFrame) -> pd.DataFrame:
    """Min/max window start time per label, in human-readable form -- surfaces the time-of-day
    confound (unauthorized is a single late block on Day 1) rather than hiding it."""
    g = window_index.groupby("label")["window_start_time_us"].agg(["min", "max", "count"])
    g["min"] = pd.to_datetime(g["min"], unit="us")
    g["max"] = pd.to_datetime(g["max"], unit="us")
    return g


if __name__ == "__main__":
    from pathlib import Path

    from ml.data_pipeline.decode_csi import REPO_ROOT

    index_path = REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"
    window_index = pd.read_csv(index_path)

    print("=== time-of-day confound check ===")
    print(time_block_report(window_index))
    print()

    print("=== session-disjoint 5-fold sanity check ===")
    for i, (train_idx, test_idx) in enumerate(session_disjoint_kfold(window_index, n_splits=5)):
        assert_no_group_leakage(window_index, train_idx, test_idx, "session_dir")
        print(f"  fold {i}: train={len(train_idx)} test={len(test_idx)} (no session leakage)")

    print()
    print("=== leave-one-unauthorized-person-out ===")
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(window_index):
        assert_no_group_leakage(window_index, train_idx, test_idx, "session_dir")
        print(f"  held out '{held_out}': train={len(train_idx)} test={len(test_idx)}")

    print()
    day_split = day_disjoint_split(window_index)
    print(f"=== day-disjoint split: {'not available yet (Day-1 only)' if day_split is None else day_split[0]} ===")
