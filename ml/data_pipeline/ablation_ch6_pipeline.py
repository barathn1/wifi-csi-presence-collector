"""Configurable channel-6 preprocessing pipeline for the cross-day preprocessing ablation study
(`ml/training/run_ablation_ch6_preprocessing.py`), covering all three labels (authorized/
unauthorized/none) so FAR-for-unauthorized and FAR-for-empty/none can both be measured, not just
authorized-vs-authorized identity accuracy.

Reuses, unchanged, the parts of `bilstm_ch6_pipeline.py` that have nothing to do with which
outlier-removal/smoothing/normalization method is selected:
- `select_dominant_packets` (MAC-pair + PHY-config validation -- diagram steps 1+2).
- `session_channel` / the channel-11-on-2026-09-15 exclusion.
- the common-packet-rate time resampling (`time_resample.py`), so "200 packets" means the same
  duration on every session/day regardless of which preprocessing config is active.

What's new here is only: (a) the manifest includes unauthorized/none, not just authorized, and
(b) `denoise_session` dispatches to `ablation_preprocessing.apply_outlier_removal`/`apply_smoothing`
per a `PreprocessConfig` instead of always running Hampel+Butterworth, and caches each config's
result under its own subdirectory so the 8 experiments never collide or get silently reused from
the wrong config.

Normalization is NOT applied in `denoise_session` -- it needs training-set-only statistics, which
aren't known until a train/test split is chosen. `compute_train_normalization_stats` computes them
from the training windows' own cached (already denoised+smoothed) sessions, and `make_normalizer`
turns them into a `CsiWindowDataset` calibration callable, frozen and reused unchanged on test data
(never recomputed on -- or leaking from -- validation/test days).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.ablation_preprocessing import PreprocessConfig, apply_outlier_removal, apply_smoothing
from ml.data_pipeline.bilstm_ch6_pipeline import (
    N_SUB_EXPECTED,
    STRIDE_PACKETS,
    TARGET_CHANNEL,
    TARGET_DATES,
    WINDOW_PACKETS,
    select_dominant_packets,
    session_channel,
)
from ml.data_pipeline.decode_csi import REPO_ROOT, decode_session_by_bucket, load_session
from ml.data_pipeline.time_resample import compute_target_rate_hz, resample_time_axis, session_native_rate_hz
from ml.data_pipeline.windowing import load_manifest

TARGET_LABELS = ["authorized", "unauthorized", "none"]
CACHE_ROOT = REPO_ROOT / "ml/data_pipeline/cache/ablation_ch6"


def build_ch6_manifest_all_labels() -> pd.DataFrame:
    """authorized + unauthorized + none rows for 2026-09-15/16/17, channel-6 sessions only -- the
    taskF_identity_or_nonauth (anjali/barath/non_auth) superset of `bilstm_ch6_pipeline`'s
    authorized-only manifest."""
    manifest = load_manifest()
    date_mask = manifest["session_dir"].apply(lambda s: any(f"/{d}/" in s for d in TARGET_DATES))
    rows = manifest[date_mask & manifest["label"].isin(TARGET_LABELS)].copy()
    rows["channel_primary"] = rows["session_dir"].apply(session_channel)
    dropped = rows[rows["channel_primary"] != TARGET_CHANNEL]
    if len(dropped):
        print(f"excluding {len(dropped)} non-channel-{TARGET_CHANNEL} session(s): "
              f"{dropped['session_dir'].tolist()} (channel={dropped['channel_primary'].tolist()})")
    ch6 = rows[rows["channel_primary"] == TARGET_CHANNEL].drop(columns=["channel_primary"])
    ch6["date"] = ch6["session_dir"].apply(lambda s: s.split("/")[1])
    return ch6.reset_index(drop=True)


def _outlier_key(config: PreprocessConfig) -> str:
    """Cache key covering only the outlier-removal stage's own parameters -- E2/E5/E7 all run
    identical Hampel params and E4/E6/E8 all run identical moving-IQR params, so this lets those
    experiments share one (comparatively expensive) outlier-removal pass instead of each redoing it
    from scratch under their own `config.name` directory."""
    if config.outlier_method is None:
        return "raw"
    return f"{config.outlier_method}_w{config.outlier_window}_k{config.outlier_k}"


def _outlier_removed_session(session_dir_rel: str, config: PreprocessConfig, force: bool = False) -> Path:
    """Validate packet metadata, decode, then apply only `config`'s outlier-removal stage. Cached by
    `_outlier_key`, not `config.name`, so experiments with identical outlier params share this."""
    safe_name = session_dir_rel.replace("/", "__")
    out_dir = CACHE_ROOT / "outlier_removed" / _outlier_key(config)
    out_path = out_dir / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    dominant_idx = select_dominant_packets(session.npz, session.metadata["board_mac"])
    sub_npz = {k: v[dominant_idx] for k, v in session.npz.items() if k != "csi_flat"}
    sub_npz["csi_flat"] = session.npz["csi_flat"]  # unchanged -- csi_offset (re-selected) still indexes into it
    buckets = decode_session_by_bucket(sub_npz)
    assert len(buckets) == 1, f"{session_dir_rel}: metadata validation should leave exactly one csi_len bucket"
    bucket = next(iter(buckets.values()))
    if bucket["n_subcarriers"] != N_SUB_EXPECTED:
        raise ValueError(f"{session_dir_rel}: validated packets have {bucket['n_subcarriers']} "
                          f"subcarriers, expected {N_SUB_EXPECTED}")

    device_time_us = sub_npz["device_time_us"].astype(np.int64)
    rssi = sub_npz["rssi"].astype(np.float32)
    native_rate_hz = session_native_rate_hz(device_time_us)

    amp = bucket["amplitude"].astype(np.float32)
    phase = np.unwrap(bucket["phase"].astype(np.float32), axis=0)  # avoid wrap artifacts before any temporal op

    amp = apply_outlier_removal(amp, config)
    phase = apply_outlier_removal(phase, config)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, amplitude=amp.astype(np.float32), phase=phase.astype(np.float32), rssi=rssi,
              device_time_us=device_time_us, native_rate_hz=np.float64(native_rate_hz))
    return out_path


def denoise_session(session_dir_rel: str, config: PreprocessConfig, force: bool = False) -> Path:
    """Outlier-removed (see `_outlier_removed_session`) plus `config`'s smoothing stage (amplitude
    and phase both, though only amplitude is used by the ablation's amplitude-only models). Cached
    per full `config.name` since smoothing on/off and its window size do vary per experiment even
    when the outlier-removal stage is shared."""
    safe_name = session_dir_rel.replace("/", "__")
    out_dir = CACHE_ROOT / "denoised" / config.name
    out_path = out_dir / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    with np.load(_outlier_removed_session(session_dir_rel, config, force=force)) as d:
        amp, phase = d["amplitude"], d["phase"]
        rssi, device_time_us, native_rate_hz = d["rssi"], d["device_time_us"], float(d["native_rate_hz"])

    amp = apply_smoothing(amp, config, rate_hz=native_rate_hz)
    phase = apply_smoothing(phase, config, rate_hz=native_rate_hz)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, amplitude=amp.astype(np.float32), phase=phase.astype(np.float32), rssi=rssi,
              device_time_us=device_time_us, native_rate_hz=np.float64(native_rate_hz))
    return out_path


def compute_dataset_target_rate_hz(manifest: pd.DataFrame, config: PreprocessConfig) -> float:
    rates = []
    for session_dir in manifest["session_dir"]:
        with np.load(denoise_session(session_dir, config)) as d:
            rates.append(float(d["native_rate_hz"]))
    target = compute_target_rate_hz(rates)
    return target


def time_normalize_session(session_dir_rel: str, config: PreprocessConfig, target_rate_hz: float,
                            force: bool = False) -> Path:
    safe_name = session_dir_rel.replace("/", "__")
    out_dir = CACHE_ROOT / "timenorm" / config.name / f"{target_rate_hz:.2f}hz"
    out_path = out_dir / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    with np.load(denoise_session(session_dir_rel, config)) as d:
        amp, phase, rssi, device_time_us = d["amplitude"], d["phase"], d["rssi"], d["device_time_us"]
    amp_r, phase_r, rssi_r, device_time_us_r = resample_time_axis(amp, phase, device_time_us, rssi, target_rate_hz)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, amplitude=amp_r, phase=phase_r, rssi=rssi_r, device_time_us=device_time_us_r)
    return out_path


def build_window_index(manifest: pd.DataFrame, config: PreprocessConfig, target_rate_hz: float,
                        window_packets: int = WINDOW_PACKETS, stride_packets: int = STRIDE_PACKETS) -> pd.DataFrame:
    """One row per fixed-length window, same schema as `bilstm_ch6_pipeline.build_window_index` plus
    `label`/`person_id` retained for every one of the three labels (needed to compute FAR-unauthorized
    and FAR-none separately at evaluation time, after taskF collapses both into `non_auth`)."""
    rows = []
    for _, row in manifest.iterrows():
        cache_path = time_normalize_session(row["session_dir"], config, target_rate_hz)
        with np.load(cache_path, mmap_mode="r") as d:
            n = d["amplitude"].shape[0]
            device_time_us = d["device_time_us"]
            if n < window_packets:
                continue
            person_id = "" if pd.isna(row.get("person_id")) else row["person_id"]
            for start in range(0, n - window_packets + 1, stride_packets):
                end = start + window_packets
                rows.append({
                    "cache_path": str(cache_path), "start": start, "end": end,
                    "session_dir": row["session_dir"], "label": row["label"], "person_id": person_id,
                    "motion": "" if pd.isna(row.get("motion")) else row["motion"],
                    "date": row["date"], "window_start_time_us": int(device_time_us[start]),
                })
    index = pd.DataFrame(rows)
    print(f"[{config.name}] built {len(index)} windows ({window_packets} packets = "
          f"{window_packets / target_rate_hz:.2f}s, stride {stride_packets} packets)")
    return index


def build_or_load_window_index(manifest: pd.DataFrame, config: PreprocessConfig, target_rate_hz: float) -> pd.DataFrame:
    cache_csv = CACHE_ROOT / "window_index" / f"{config.name}_{target_rate_hz:.2f}hz_n{len(manifest)}.csv"
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    index = build_window_index(manifest, config, target_rate_hz)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(cache_csv, index=False)
    return index


def compute_train_normalization_stats(train_window_index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-subcarrier (mean, std) pooled over every packet of every TRAINING session referenced by
    `train_window_index` -- not per-window, and never touching any test-day session. Frozen and
    reused unchanged on validation/test days by `make_normalizer`."""
    total = np.zeros(N_SUB_EXPECTED, dtype=np.float64)
    total_sq = np.zeros(N_SUB_EXPECTED, dtype=np.float64)
    count = 0
    for cache_path in train_window_index["cache_path"].unique():
        with np.load(cache_path, mmap_mode="r") as d:
            amp = np.asarray(d["amplitude"], dtype=np.float64)
        total += amp.sum(axis=0)
        total_sq += (amp ** 2).sum(axis=0)
        count += amp.shape[0]
    mean = total / count
    var = total_sq / count - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return mean.astype(np.float32), std.astype(np.float32)


def make_normalizer(mean: np.ndarray, std: np.ndarray):
    """Calibration callable for `CsiWindowDataset`: normalizes amplitude with the frozen
    training-set stats, passes phase through untouched (only amplitude feeds the ablation's models)."""
    eps = 1e-6

    def _normalize(amp: np.ndarray, phase: np.ndarray, row) -> tuple[np.ndarray, np.ndarray]:
        amp_n = (amp - mean) / (std + eps)
        return amp_n.astype(np.float32), phase
    return _normalize
