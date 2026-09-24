"""Builds the channel-6-only dataset every model in ml_2 trains on, using ONLY the fresh ml_2/data/*
pipeline (no ml.data_pipeline import). Each (session, receiver) unit is its own row -- multi-receiver
sessions (2026-09-21/22) contribute one row per ESP, not a single collapsed 'representative' receiver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml_2.data.calibration import Baseline, make_calibration_fn, per_receiver_day_baselines
from ml_2.data.features import build_feature_matrix, feature_names
from ml_2.data.manifest import build_channel6_manifest
from ml_2.data.windowing import CACHE_DIR, cache_all, build_window_index

N_SUBCARRIERS = 64  # legacy (LLTF) subcarriers only -- see windowing.py's N_LEGACY_SUBCARRIERS comment;
# every channel-6 row's raw dominant bucket is 256 bytes/128 subcarriers (LLTF+HT-LTF merged), sliced
# down to the first 64 at cache time.


@dataclass
class Channel6Dataset:
    manifest: pd.DataFrame
    window_index: pd.DataFrame
    calibration: callable


def load_channel6_dataset(window_packets: int = 200, stride_packets: int | None = None) -> Channel6Dataset:
    manifest = build_channel6_manifest()
    print("caching (session, receiver) decoded arrays...")
    manifest = cache_all(manifest)
    window_index = build_window_index(manifest, window_packets=window_packets, stride_packets=stride_packets)

    print("computing per-(date, receiver) empty-room baselines...")
    baselines = per_receiver_day_baselines(manifest)
    print(f"  {len(baselines)} baselines: {sorted(baselines.keys())}")
    calibration = make_calibration_fn(baselines)
    return Channel6Dataset(manifest=manifest, window_index=window_index, calibration=calibration)


_FEATURE_CACHE: dict[int, np.ndarray] = {}
_DISK_CACHE_PATH = CACHE_DIR.parent / "features_channel6.npy"


def feature_matrix_for(window_index: pd.DataFrame, dataset: Channel6Dataset) -> np.ndarray:
    """In-memory cache keyed by window_index identity (reused across folds within one process) PLUS a
    disk cache (reused across the separate SVM/GBM/CUSUM/generative-hard-negative subprocesses, which
    each start a fresh Python process and would otherwise recompute this ~178k-window feature matrix
    from scratch every time -- the actual bottleneck once run_all.py runs each model as its own
    subprocess). Assumes window_index doesn't change shape/order between runs of the same dataset
    build; if that build ever becomes non-deterministic, delete the .npy to force a rebuild."""
    key = id(window_index)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]
    if _DISK_CACHE_PATH.exists():
        cached = np.load(_DISK_CACHE_PATH)
        if cached.shape[0] == len(window_index):
            _FEATURE_CACHE[key] = cached
            return cached
        print(f"  disk feature cache shape {cached.shape} doesn't match window_index len "
              f"{len(window_index)} -- recomputing")
    X = build_feature_matrix(window_index, calibration=dataset.calibration)
    _FEATURE_CACHE[key] = X
    _DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(_DISK_CACHE_PATH, X)
    return X


FEATURE_NAMES = feature_names(N_SUBCARRIERS)
