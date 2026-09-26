"""Cross-day split: leave-one-day-out over sessions. No window ever straddles train/test since we split
by date -- overlapping windows are already confined to one session, and sessions are confined to one date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def leave_one_day_out(window_table: pd.DataFrame):
    for held_out_date in sorted(window_table["date"].unique()):
        test_mask = (window_table["date"] == held_out_date).values
        yield held_out_date, np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def assert_no_session_leakage(window_table: pd.DataFrame, train_idx, test_idx) -> None:
    train_sessions = set(window_table.iloc[train_idx]["session_dir"])
    test_sessions = set(window_table.iloc[test_idx]["session_dir"])
    overlap = train_sessions & test_sessions
    assert not overlap, f"session leakage: {overlap}"
