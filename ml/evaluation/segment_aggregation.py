"""Segment-level score aggregation: average a model's per-window score over several consecutive,
overlapping 1s windows before making a decision, instead of deciding from a single ~1s window.

Matches ARGUS's (arXiv:2608.14670) finding that this is a "free" accuracy gain -- averaging softmax
over 19 overlapping 6s windows raised their Top-1 from 78.88% to 84.85% with zero retraining, same
trained model. No packet-level or single-window decision is used anywhere in this pipeline already
(the smallest unit is a 200-packet/~1s window, see windowing.py) -- this adds a longer, multi-window
decision horizon on top of that, matching a "watch someone for 5-10 seconds" style deployment.

Windows are built with 50% overlap (stride=100, window_packets=200 by default), so N consecutive
windows from the same session span roughly `(N-1)*stride + window_packets` packets:
  N=10 -> ~1100 packets =~5s at this data's ~217 Hz average rate
  N=20 -> ~2100 packets =~10s
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_scores_by_session(test_index: pd.DataFrame, scores: np.ndarray, n_windows: int = 10) -> pd.DataFrame:
    """Non-overlapping groups of N consecutive windows within each session -> one averaged score each.
    test_index must be in the same row order as `scores` (both indexed 0..len-1)."""
    df = test_index.reset_index(drop=True).copy()
    df["score"] = scores

    segments = []
    for session_dir, group in df.groupby("session_dir", sort=False):
        group = group.sort_values("start").reset_index(drop=True)
        for i in range(0, len(group) - n_windows + 1, n_windows):
            seg = group.iloc[i:i + n_windows]
            segments.append({
                "session_dir": session_dir,
                "label": seg["label"].iloc[0],
                "person_id": seg["person_id"].iloc[0],
                "aggregated_score": seg["score"].mean(),
                "n_windows": len(seg),
                "span_packets": int(seg["end"].iloc[-1] - seg["start"].iloc[0]),
            })
    return pd.DataFrame(segments)
