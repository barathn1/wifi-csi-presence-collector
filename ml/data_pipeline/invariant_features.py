"""Permutation-invariant, subcarrier-identity-free features -- see ml/reports/day2_next_steps.md item 9
and [[project-day2-cross-channel-root-cause]] in memory. `features.py`'s per-subcarrier mean/std/skew/
kurt are indexed BY subcarrier position, which is exactly the thing that's broken across Day1 (186
subcarriers, 2402-2442MHz) vs Day2 (128 subcarriers, 2452-2472MHz) -- "subcarrier 64" means a different
physical frequency on each day. These features instead describe the SHAPE of the amplitude/phase
distribution across the window without referencing which subcarrier index held which value, so they're
at least candidate-comparable across two sessions with a different subcarrier count/mapping entirely
(not proven to transfer better -- that's an empirical question for whoever runs this against the
per-subcarrier features, see the ranked-queue framing in day2_next_steps.md; this module only builds
the candidate feature set).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.windowing import iter_windows

EPS = 1e-6
PERCENTILES = (10, 25, 50, 75, 90)
TOP_K_EIGENVALUES = 3


def _spectral_entropy(profile: np.ndarray) -> float:
    """Shannon entropy of the per-subcarrier mean-amplitude profile, treated as a distribution over
    (unordered) subcarriers -- high entropy = energy spread evenly across the band, low = concentrated
    in a few subcarriers. Depends only on the multiset of values, not which index held which."""
    p = np.abs(profile)
    p = p / (p.sum() + EPS)
    return float(-(p * np.log(p + EPS)).sum())


def _top_k_eigenvalue_fractions(window: np.ndarray, k: int = TOP_K_EIGENVALUES) -> np.ndarray:
    """window: (T, n_sub). Eigenvalues of the (n_sub, n_sub) covariance matrix, as a fraction of total
    variance (trace) -- describes how many effective independent modes of variation the window has and
    how concentrated they are, without depending on subcarrier identity (a permutation of subcarrier
    order permutes the covariance matrix by a similarity transform, which leaves eigenvalues unchanged)."""
    cov = np.cov(window, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    total = eigvals.sum() + EPS
    top = eigvals[::-1][:k] / total
    if len(top) < k:
        top = np.pad(top, (0, k - len(top)))
    return top


def _lag1_autocorr_mean(window: np.ndarray) -> float:
    """Per-subcarrier lag-1 autocorrelation in time, averaged across subcarriers -- a temporal-
    smoothness measure, invariant to subcarrier identity because it's averaged over all of them."""
    x = window - window.mean(axis=0)
    num = (x[:-1] * x[1:]).sum(axis=0)
    den = (x ** 2).sum(axis=0) + EPS
    return float((num / den).mean())


def _channel_invariant_stats(arr: np.ndarray) -> np.ndarray:
    """arr: (T, n_sub) for one channel (amplitude or phase) -> 1D vector of invariant stats."""
    flat = arr.ravel()
    pct = np.percentile(flat, PERCENTILES)
    mean, std = flat.mean(), flat.std()
    diff = flat - mean
    skew = (diff ** 3).mean() / (std ** 3 + EPS)
    kurt = (diff ** 4).mean() / (std ** 4 + EPS) - 3.0
    profile = arr.mean(axis=0)
    entropy = _spectral_entropy(profile)
    eig_fracs = _top_k_eigenvalue_fractions(arr)
    autocorr = _lag1_autocorr_mean(arr)
    return np.concatenate([pct, [skew, kurt, entropy, autocorr], eig_fracs])


def invariant_feature_names() -> list[str]:
    names = []
    for channel in ("amp", "phase"):
        names += [f"{channel}_pct{p}" for p in PERCENTILES]
        names += [f"{channel}_skew", f"{channel}_kurt", f"{channel}_spectral_entropy", f"{channel}_lag1_autocorr"]
        names += [f"{channel}_eigfrac{i}" for i in range(TOP_K_EIGENVALUES)]
    names += ["rssi_mean", "rssi_std"]
    return names


def extract_invariant_window_features(amplitude: np.ndarray, phase: np.ndarray, rssi: np.ndarray) -> np.ndarray:
    parts = [_channel_invariant_stats(amplitude), _channel_invariant_stats(phase),
             np.array([rssi.mean(), rssi.std()])]
    features = np.concatenate(parts).astype(np.float32)
    return np.nan_to_num(features, nan=0.0)


def build_invariant_feature_matrix(window_index: pd.DataFrame, calibration=None) -> np.ndarray:
    rows = []
    for amp, phase, rssi, row in iter_windows(window_index):
        if calibration is not None:
            amp, phase = calibration(amp, phase, row)
        rows.append(extract_invariant_window_features(amp, phase, rssi))
    return np.stack(rows)


def build_and_cache_invariant_features(window_index_path, out_path, calibration=None) -> np.ndarray:
    window_index = pd.read_csv(window_index_path)
    X = build_invariant_feature_matrix(window_index, calibration=calibration)
    np.save(out_path, X)
    print(f"cached {X.shape} -> {out_path}")
    return X


if __name__ == "__main__":
    import argparse

    from ml.data_pipeline.decode_csi import REPO_ROOT

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default=str(REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"))
    p.add_argument("--smoke-test", action="store_true", help="quick 20-window check instead of full build")
    args = p.parse_args()

    if args.smoke_test:
        window_index = pd.read_csv(args.index).sample(20, random_state=0)
        X = build_invariant_feature_matrix(window_index)
        names = invariant_feature_names()
        print(f"feature matrix: {X.shape} (expected {len(names)} columns: {len(names)})")
        assert X.shape[1] == len(names)
        print("OK -- no NaNs:", not np.isnan(X).any())
    else:
        cache_dir = REPO_ROOT / "ml/data_pipeline/cache"
        index_name = Path(args.index).stem
        print("building invariant feature matrix...")
        build_and_cache_invariant_features(args.index, cache_dir / f"features_invariant_{index_name}.npy")
