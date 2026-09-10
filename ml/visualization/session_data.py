"""Shared session decode/downsample logic used by both the (now data-free) player.html template and
player_server.py, which builds this data ON DEMAND when a browser asks for a specific session -- see
player_server.py's module docstring for why nothing is pre-built ahead of time anymore.
"""
from __future__ import annotations

import numpy as np

from ml.data_pipeline.windowing import DOMINANT_CSI_LEN, cache_session

TARGET_RATE_HZ = 25.0  # downsampled time resolution target -- see session_width()
MIN_WIDTH, MAX_WIDTH = 400, 8000


def session_width(duration_s: float) -> int:
    """Time-bin count targeting ~25 samples/sec of downsampled resolution regardless of session
    length, capped so a response stays bounded for very long sessions -- a single fixed width for
    every session either wastes resolution on short ones or destroys it on long ones."""
    return int(np.clip(round(TARGET_RATE_HZ * duration_s), MIN_WIDTH, MAX_WIDTH))


def load_channel(session_dir_rel: str, channel: str) -> np.ndarray:
    cache_path = cache_session(session_dir_rel, DOMINANT_CSI_LEN)
    with np.load(cache_path) as d:
        return d["amplitude"] if channel == "amplitude" else d["phase"]


def downsample_time(values: np.ndarray, width: int) -> np.ndarray:
    """(n_packets, n_sub) -> (n_sub, width) mean-pooled along time."""
    n_packets = values.shape[0]
    width = min(width, n_packets)
    edges = np.linspace(0, n_packets, width + 1).astype(int)
    out = np.empty((values.shape[1], width))
    for i in range(width):
        s, e = edges[i], max(edges[i] + 1, edges[i + 1])
        out[:, i] = values[s:e].mean(axis=0)
    return out


def build_session_grid(session_dir_rel: str, channel: str, duration_s: float) -> np.ndarray:
    """(n_sub, width) downsampled grid for one session+channel, decoded fresh every call -- cheap
    after the first request for a given session, since cache_session() caches the raw decode."""
    values = load_channel(session_dir_rel, channel)
    return downsample_time(values, session_width(duration_s))
