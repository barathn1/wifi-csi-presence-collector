"""Configurable preprocessing primitives for the channel-6 cross-day ablation study
(`ml/training/run_ablation_ch6_preprocessing.py`). Every stage is opt-in and independently
switchable, matching the requested flow:

    IQ -> amplitude
      -> OUTLIER REMOVAL (none / hampel / iqr / moving_iqr) -- replaces flagged points,
         never drops the packet, so every array keeps its original length
      -> TEMPORAL SMOOTHING (none / moving_average)
      -> NORMALIZATION (none / train-only mean/std, applied by the caller once train-set
         stats are known -- see ablation_ch6_pipeline.compute_train_normalization_stats)

The Hampel filter is NOT reimplemented here -- it's reused verbatim from
`paper_2507_12854_preprocessing.hampel_filter`, the one existing, paper-grounded implementation
in this repo (see that module's docstring). IQR and moving-window-IQR are new: no prior
implementation of either existed anywhere in this repo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from ml.data_pipeline.paper_2507_12854_preprocessing import butterworth_lowpass, hampel_filter


@dataclass(frozen=True)
class PreprocessConfig:
    name: str
    outlier_method: str | None = None   # None | "hampel" | "iqr" | "moving_iqr"
    outlier_window: int = 15            # sliding window (packets) for "hampel"/"moving_iqr"
    outlier_k: float = 3.0              # hampel: n_sigmas (MAD units); iqr/moving_iqr: IQR multiplier
    smoothing: bool = False             # smoothing on/off
    smoothing_window: int = 5
    smoothing_method: str = "moving_average"  # "moving_average" | "butterworth"
    butterworth_cutoff_hz: float = 10.0       # only used when smoothing_method == "butterworth"
    butterworth_order: int = 5
    normalize: bool = False             # train-set mean/std normalization on/off


def remove_outliers_iqr(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Method B: global per-subcarrier IQR fence over the whole session (axis 0 = time). Flagged
    points are replaced with that subcarrier's session median -- the packet is never dropped."""
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    med = np.median(x, axis=0)
    is_outlier = (x < lower) | (x > upper)
    return np.where(is_outlier, med, x)


def remove_outliers_moving_iqr(x: np.ndarray, window: int = 15, k: float = 3.0) -> np.ndarray:
    """Method C: same IQR fence as `remove_outliers_iqr`, but the quartiles/median are recomputed
    per-subcarrier from a local sliding temporal window (like the Hampel filter's window, but using
    IQR bounds instead of MAD) rather than the whole session -- adapts to slow drift over a session
    instead of one fixed global fence. Replaces flagged points with the local window median.

    Implemented via pandas' vectorized rolling quantile (`center=True`, `min_periods=1` so edges use
    a smaller, still-centered-as-possible window instead of NaN) rather than a per-timestep Python
    loop -- a naive loop calling `np.percentile` per timestep took 20-90s for a single ~30-60k-packet
    session (measured on this project's own channel-6 data), which multiplies out to hours across 81
    sessions x 3 experiments (E4/E6/E8) that all share these exact outlier parameters."""
    df = pd.DataFrame(x)
    roll = df.rolling(window=window, center=True, min_periods=1)
    q1, q3, med = roll.quantile(0.25), roll.quantile(0.75), roll.median()
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    is_outlier = (df < lower) | (df > upper)
    return df.where(~is_outlier, med).to_numpy(dtype=x.dtype)


def moving_average_smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Per-subcarrier centered moving average along axis 0 (time), edge-replicated so the output
    keeps the input's length. `window` is forced odd so the average is exactly centered."""
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    return uniform_filter1d(x, size=window, axis=0, mode="nearest").astype(x.dtype)


def apply_outlier_removal(x: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if config.outlier_method is None:
        return x
    if config.outlier_method == "hampel":
        return hampel_filter(x, window=config.outlier_window, n_sigmas=config.outlier_k)
    if config.outlier_method == "iqr":
        return remove_outliers_iqr(x, k=config.outlier_k)
    if config.outlier_method == "moving_iqr":
        return remove_outliers_moving_iqr(x, window=config.outlier_window, k=config.outlier_k)
    raise ValueError(f"unknown outlier_method: {config.outlier_method}")


def apply_smoothing(x: np.ndarray, config: PreprocessConfig, rate_hz: float | None = None) -> np.ndarray:
    if not config.smoothing:
        return x
    if config.smoothing_method == "moving_average":
        return moving_average_smooth(x, window=config.smoothing_window)
    if config.smoothing_method == "butterworth":
        if rate_hz is None:
            raise ValueError("butterworth smoothing requires rate_hz (the session's native packet rate)")
        return butterworth_lowpass(x, fs=rate_hz, cutoff=config.butterworth_cutoff_hz, order=config.butterworth_order)
    raise ValueError(f"unknown smoothing_method: {config.smoothing_method}")


# The 8 experiments requested for the ablation study, same model/split/sequence-length for all of
# them -- only these preprocessing choices vary.
EXPERIMENTS: dict[str, PreprocessConfig] = {
    "E1_raw": PreprocessConfig("E1_raw"),
    "E2_hampel": PreprocessConfig("E2_hampel", outlier_method="hampel"),
    "E3_iqr": PreprocessConfig("E3_iqr", outlier_method="iqr"),
    "E4_moving_iqr": PreprocessConfig("E4_moving_iqr", outlier_method="moving_iqr"),
    "E5_hampel_smooth": PreprocessConfig("E5_hampel_smooth", outlier_method="hampel", smoothing=True),
    "E6_moving_iqr_smooth": PreprocessConfig("E6_moving_iqr_smooth", outlier_method="moving_iqr", smoothing=True),
    "E7_hampel_smooth_norm": PreprocessConfig(
        "E7_hampel_smooth_norm", outlier_method="hampel", smoothing=True, normalize=True),
    "E8_moving_iqr_smooth_norm": PreprocessConfig(
        "E8_moving_iqr_smooth_norm", outlier_method="moving_iqr", smoothing=True, normalize=True),
    # arXiv:2507.12854's exact noise-reduction recipe (Hampel window=15/n_sigmas=3, then 5th-order
    # Butterworth low-pass cutoff=10Hz) -- same hampel_filter/butterworth_lowpass this repo already
    # uses in bilstm_ch6_pipeline.denoise_session (taskB, authorized-only), now run through this
    # harness's taskF (anjali/barath/non_auth) manifest + leave-one-day-out split + FAR metrics so
    # it's directly comparable to E1-E8 on the same footing. Deliberately excludes the paper's
    # temporal-mean-reduction (halving) and phase-endpoint calibration: the former would change each
    # config's effective window duration (breaking the "same 200-packet window everywhere" premise
    # this ablation depends on), the latter only touches phase, which none of these amplitude-only
    # models read.
    "E9_paper_hampel_butterworth": PreprocessConfig(
        "E9_paper_hampel_butterworth", outlier_method="hampel", outlier_window=15, outlier_k=3.0,
        smoothing=True, smoothing_method="butterworth", butterworth_cutoff_hz=10.0, butterworth_order=5),
    "E10_paper_hampel_butterworth_norm": PreprocessConfig(
        "E10_paper_hampel_butterworth_norm", outlier_method="hampel", outlier_window=15, outlier_k=3.0,
        smoothing=True, smoothing_method="butterworth", butterworth_cutoff_hz=10.0, butterworth_order=5,
        normalize=True),
}
