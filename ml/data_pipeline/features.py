"""Handcrafted per-window statistical features -- the classical-ML baseline's input, and also what the
visualization scripts (class_fingerprint, effect_size_heatmap) plot instead of raw CSI traces.

Per subcarrier, per channel (amplitude, phase): mean, std, skew, kurtosis over the window's time axis.
Conceptually the same idea as ARGUS's "statgram" (arXiv:2608.14670) -- a compact statistical map instead
of raw sequences -- kept deliberately simple here (no binning/patches) since Day 1 is a small dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.csi_resample import TARGET_SUBCARRIERS
from ml.data_pipeline.windowing import iter_windows

EPS = 1e-6


def _moments(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """mean/std/skew/excess-kurtosis over axis 0, by hand -- ~30% faster than scipy.stats.skew/kurtosis
    at this call volume (~30k windows x 2 channels), since scipy's generic path re-validates/re-derives
    per call. Matches scipy's default (Fisher/excess kurtosis, population moments)."""
    mean = arr.mean(axis=0)
    diff = arr - mean
    std = np.sqrt((diff * diff).mean(axis=0))
    skew = (diff ** 3).mean(axis=0) / (std ** 3 + EPS)
    kurt = (diff ** 4).mean(axis=0) / (std ** 4 + EPS) - 3.0
    return mean, std, skew, kurt


def feature_slices(n_subcarriers: int = TARGET_SUBCARRIERS) -> dict[str, slice]:
    """Column ranges within the flat feature vector, matching feature_names()'s block order."""
    slices, i = {}, 0
    for channel in ("amp", "phase"):
        for stat in ("mean", "std", "skew", "kurt"):
            slices[f"{channel}_{stat}"] = slice(i, i + n_subcarriers)
            i += n_subcarriers
    slices["rssi_mean"] = slice(i, i + 1)
    slices["rssi_std"] = slice(i + 1, i + 2)
    return slices


def feature_names(n_subcarriers: int = TARGET_SUBCARRIERS) -> list[str]:
    names = []
    for channel in ("amp", "phase"):
        for stat in ("mean", "std", "skew", "kurt"):
            names += [f"{channel}_{stat}_sc{i}" for i in range(n_subcarriers)]
    names += ["rssi_mean", "rssi_std"]
    return names


def extract_window_features(amplitude: np.ndarray, phase: np.ndarray, rssi: np.ndarray) -> np.ndarray:
    """amplitude/phase: (window_packets, n_subcarriers), rssi: (window_packets,) -> 1D feature vector."""
    parts = []
    for arr in (amplitude, phase):
        parts.extend(_moments(arr))
    parts.append(np.array([rssi.mean()]))
    parts.append(np.array([rssi.std()]))
    features = np.concatenate(parts).astype(np.float32)
    # skew/kurtosis are undefined (NaN) for subcarriers that are constant across the whole window
    # (e.g. the zero-padded leading subcarriers seen in some frames) -- no variability to describe, so 0.
    return np.nan_to_num(features, nan=0.0)


def build_feature_matrix(window_index: pd.DataFrame, calibration=None) -> np.ndarray:
    """calibration: optional callable (amplitude, phase, row) -> (amplitude, phase), e.g. a Variant-A
    baseline's apply function, applied before feature extraction."""
    rows = []
    for amp, phase, rssi, row in iter_windows(window_index):
        if calibration is not None:
            amp, phase = calibration(amp, phase, row)
        rows.append(extract_window_features(amp, phase, rssi))
    return np.stack(rows)


def build_and_cache_features(window_index_path, out_path, calibration=None) -> np.ndarray:
    window_index = pd.read_csv(window_index_path)
    X = build_feature_matrix(window_index, calibration=calibration)
    np.save(out_path, X)
    print(f"cached {X.shape} -> {out_path}")
    return X


if __name__ == "__main__":
    import argparse

    from ml.data_pipeline.calibration import apply_variant_a, compute_day_baseline
    from ml.data_pipeline.decode_csi import REPO_ROOT
    from ml.data_pipeline.windowing import load_manifest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default=str(REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"))
    p.add_argument("--smoke-test", action="store_true", help="quick 20-window check instead of full build")
    args = p.parse_args()

    if args.smoke_test:
        window_index = pd.read_csv(args.index).sample(20, random_state=0)
        X = build_feature_matrix(window_index)
        names = feature_names()
        print(f"feature matrix: {X.shape} (expected {len(names)} columns: {len(names)})")
        assert X.shape[1] == len(names)
        print("OK -- no NaNs:", not np.isnan(X).any())
    else:
        cache_dir = REPO_ROOT / "ml/data_pipeline/cache"
        index_name = Path(args.index).stem

        print("building RAW feature matrix...")
        build_and_cache_features(args.index, cache_dir / f"features_raw_{index_name}.npy")

        manifest = load_manifest()
        window_index = pd.read_csv(args.index)
        mode = "native" if index_name.endswith("_native") else "resampled"
        baselines = {date: compute_day_baseline(manifest, date, mode=mode)
                     for date in window_index["date"].unique()}

        def calib_a(amp, phase, row):
            return apply_variant_a(amp, phase, baselines[row["date"]])

        print("building Variant-A-calibrated feature matrix...")
        build_and_cache_features(args.index, cache_dir / f"features_calibA_{index_name}.npy", calibration=calib_a)
