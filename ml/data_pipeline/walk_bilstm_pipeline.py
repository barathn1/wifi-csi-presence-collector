"""Walking-only channel-6 preprocessing for the BiLSTM person-identity task (2026-09-15/16/17,
anjali vs barath), built fresh to the spec: per-subcarrier Hampel -> Butterworth lowpass -> time-based
sliding window with fixed-length interpolation -> [N, 400, 256] per session, deferring top-variance
subcarrier selection and per-window normalization to the training script (see below for why).

Two corrections vs. a naive from-scratch reading of the spec, both confirmed against this repo's own
prior packet-level analysis rather than assumed:

1. Link/PHY filter: NOT `mcs==0 & stbc==0 & cwb==0`. That combo selects the overheard laptop<->router
   stimulus traffic (a different physical propagation path), not the router->ESP32 link that actually
   passes through where the person is walking. `select_dominant_packets` (imported from
   `bilstm_ch6_pipeline.py`) identifies the right link by `dst_mac == board_mac` plus each session's own
   dominant (channel/cwb/stbc/mcs/csi_len) combo -- evidence-based, see that function's docstring for the
   full per-session MAC-pair breakdown that established this.
2. Subcarrier count: 128, not 52. The "128 = 64 LLTF + 64 HT-LTF, trim to 52" split is generic ESP32 CSI
   folklore that doesn't hold for this hardware's dominant channel-6 packets (verified via
   `decode_csi.py`'s round-trip check against the raw wire bytes -- no null block). Top-variance
   subcarrier selection (spec step 7) already exists precisely to auto-drop dead/furniture subcarriers,
   so it runs against the real 128 rather than a mis-derived 52.

Design choice NOT in the pasted spec: top-30 variance subcarrier selection and per-window per-subcarrier
normalization are DEFERRED to the training script, computed from TRAINING-FOLD windows only and then
frozen/applied to the held-out day. The pasted spec computes both from the whole per-day dataset inside
`preprocess_day`, but Leave-One-Day-Out CV means "the whole dataset" always includes the test day at
preprocessing time -- selecting subcarriers (or normalizing) using test-day statistics would leak
cross-day information into a cross-day evaluation. This module therefore stops at the un-reduced,
un-normalized [N, 400, 256] (128 amp + 128 phase) tensor; `select_topk_variance`/`apply_topk`/
`per_window_zscore` below are still here (same math as the spec), just meant to be called per-fold from
the training script on `train`-only windows before being applied unchanged to `test` windows.

Do not combine standing + walking data in these windows -- only `motion == "walking"` rows are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ml.data_pipeline.bilstm_ch6_pipeline import N_SUB_EXPECTED, TARGET_CHANNEL, TARGET_DATES, select_dominant_packets
from ml.data_pipeline.decode_csi import REPO_ROOT, decode_session_by_bucket, load_session
from ml.data_pipeline.paper_2507_12854_preprocessing import butterworth_lowpass, hampel_filter
from ml.data_pipeline.time_resample import session_native_rate_hz
from ml.data_pipeline.windowing import load_manifest

WINDOW_SEC = 4.0
OVERLAP = 0.5
FINAL_LEN = 400
TOP_K_SUBCARRIERS = 30
HAMPEL_WINDOW = 10
HAMPEL_N_SIGMAS = 3.0
BUTTER_CUTOFF_HZ = 20.0
BUTTER_ORDER = 4
MIN_PACKETS_PER_WINDOW = 10  # below this, a window's interpolation would be fabricating most of its points

CACHE_ROOT = REPO_ROOT / "ml/data_pipeline/cache/walk_bilstm"
DECODED_CACHE_DIR = CACHE_ROOT / "decoded"
DENOISED_CACHE_DIR = CACHE_ROOT / "denoised"
WINDOWS_CACHE_DIR = CACHE_ROOT / "windows"


def _walk_manifest_for_label(label: str) -> pd.DataFrame:
    """walking rows for 2026-09-15/16/17, channel-6 sessions only, for one manifest `label`
    (authorized/unauthorized). Drops 20260915_143748_anjali, which this repo's own channel_primary
    check found is channel 11, a mid-collection router hop -- see `bilstm_ch6_pipeline.session_channel`
    (that particular session is `authorized`, so this only ever fires for `label="authorized"`, but the
    check is label-agnostic since a future collection could hop channel on any label)."""
    manifest = load_manifest()
    date_mask = manifest["session_dir"].apply(lambda s: any(f"/{d}/" in s for d in TARGET_DATES))
    rows = manifest[date_mask & (manifest["label"] == label) & (manifest["motion"] == "walking")].copy()
    from ml.data_pipeline.bilstm_ch6_pipeline import session_channel
    rows["channel_primary"] = rows["session_dir"].apply(session_channel)
    dropped = rows[rows["channel_primary"] != TARGET_CHANNEL]
    if len(dropped):
        print(f"excluding {len(dropped)} non-channel-{TARGET_CHANNEL} walking session(s): "
              f"{dropped['session_dir'].tolist()} (channel={dropped['channel_primary'].tolist()})")
    ch6 = rows[rows["channel_primary"] == TARGET_CHANNEL].drop(columns=["channel_primary"])
    ch6["date"] = ch6["session_dir"].apply(lambda s: s.split("/")[1])
    return ch6.reset_index(drop=True)


def build_walk_manifest() -> pd.DataFrame:
    """authorized + walking rows for 2026-09-15/16/17, channel-6 sessions only (anjali/barath --
    the 2 enrolled identities the BiLSTM classifier is trained on)."""
    return _walk_manifest_for_label("authorized")


def build_unauthorized_walk_manifest() -> pd.DataFrame:
    """unauthorized + walking rows for 2026-09-15/16/17, channel-6 sessions only (divya/harshitha/
    sumanth/abdul/siva/manas/kishore -- never used for training, only for open-set false-accept
    evaluation in `ml/training/run_walk_bilstm_openset.py`)."""
    return _walk_manifest_for_label("unauthorized")


def decode_session(session_dir_rel: str, force: bool = False) -> Path:
    """Validate packet metadata (right link, one consistent PHY combo), decode IQ -> amplitude/phase,
    cache. No time reduction here -- device_time_us is kept at native resolution for accurate
    time-based windowing/interpolation downstream."""
    safe_name = session_dir_rel.replace("/", "__")
    out_path = DECODED_CACHE_DIR / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    dominant_idx = select_dominant_packets(session.npz, session.metadata["board_mac"])
    sub_npz = {k: v[dominant_idx] for k, v in session.npz.items() if k != "csi_flat"}
    sub_npz["csi_flat"] = session.npz["csi_flat"]
    buckets = decode_session_by_bucket(sub_npz)
    assert len(buckets) == 1, f"{session_dir_rel}: metadata validation should leave exactly one csi_len bucket"
    bucket = next(iter(buckets.values()))
    if bucket["n_subcarriers"] != N_SUB_EXPECTED:
        raise ValueError(f"{session_dir_rel}: validated packets have {bucket['n_subcarriers']} "
                          f"subcarriers, expected {N_SUB_EXPECTED}")

    device_time_us = sub_npz["device_time_us"].astype(np.int64)
    native_rate_hz = session_native_rate_hz(device_time_us)

    DECODED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, amplitude=bucket["amplitude"], phase=bucket["phase"], device_time_us=device_time_us,
              native_rate_hz=np.float64(native_rate_hz))
    return out_path


def sanitize_phase(phase: np.ndarray) -> np.ndarray:
    """Per-packet: unwrap across the subcarrier axis, then remove the FULL least-squares linear trend
    fit over all subcarriers (not just the two endpoints) -- spec step 3, vectorized across packets."""
    unwrapped = np.unwrap(phase, axis=1)
    n_sub = phase.shape[1]
    k = np.arange(n_sub, dtype=np.float64)
    k_centered = k - k.mean()
    denom = float(np.sum(k_centered ** 2))
    y_mean = unwrapped.mean(axis=1, keepdims=True)
    slope = (unwrapped - y_mean) @ k_centered / denom  # (n_packets,)
    intercept = y_mean[:, 0] - slope * k.mean()
    trend = slope[:, None] * k[None, :] + intercept[:, None]
    return (unwrapped - trend).astype(np.float32)


def denoise_session(session_dir_rel: str, force: bool = False) -> Path:
    """Amp/phase sanitization, per-subcarrier Hampel outlier removal, Butterworth 20Hz lowpass -- spec
    steps 3-5, applied to the real 128 subcarriers (not a mis-derived 52)."""
    safe_name = session_dir_rel.replace("/", "__")
    out_path = DENOISED_CACHE_DIR / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    with np.load(decode_session(session_dir_rel)) as d:
        amp, phase = d["amplitude"].astype(np.float32), d["phase"].astype(np.float32)
        device_time_us, native_rate_hz = d["device_time_us"], float(d["native_rate_hz"])

    phase_sanitized = sanitize_phase(phase)
    features = np.concatenate([amp, phase_sanitized], axis=1)  # (n_packets, 256): [amp(128), phase(128)]

    features = hampel_filter(features, window=HAMPEL_WINDOW, n_sigmas=HAMPEL_N_SIGMAS)
    features = butterworth_lowpass(features, fs=native_rate_hz, cutoff=BUTTER_CUTOFF_HZ, order=BUTTER_ORDER)

    DENOISED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, features=features.astype(np.float32), device_time_us=device_time_us)
    return out_path


def make_windows(features: np.ndarray, device_time_us: np.ndarray, window_sec: float = WINDOW_SEC,
                  overlap: float = OVERLAP, final_len: int = FINAL_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Time-based sliding window (real seconds, not packet counts -- robust to packet loss/rate drift)
    + interpolation to a fixed `final_len` points per window -- spec step 6. Returns
    (windows [n_windows, final_len, n_features], window_start_time_us [n_windows])."""
    t = (device_time_us - device_time_us[0]).astype(np.float64) / 1e6
    total_time = t[-1]
    step = window_sec * (1 - overlap)

    windows, starts_us = [], []
    start_t = 0.0
    while start_t + window_sec <= total_time:
        end_t = start_t + window_sec
        idx = np.flatnonzero((t >= start_t) & (t < end_t))
        if len(idx) >= MIN_PACKETS_PER_WINDOW:
            seg_t, seg = t[idx], features[idx]
            new_t = np.linspace(start_t, end_t, final_len, endpoint=False)
            interp = interp1d(seg_t, seg, axis=0, kind="linear", fill_value="extrapolate", assume_sorted=True)
            windows.append(interp(new_t).astype(np.float32))
            starts_us.append(int(device_time_us[0] + start_t * 1e6))
        start_t += step

    if not windows:
        return np.empty((0, final_len, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(windows), np.array(starts_us, dtype=np.int64)


def build_session_windows(session_dir_rel: str, force: bool = False) -> Path:
    safe_name = session_dir_rel.replace("/", "__")
    out_path = WINDOWS_CACHE_DIR / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    with np.load(denoise_session(session_dir_rel)) as d:
        features, device_time_us = d["features"], d["device_time_us"]
    windows, starts_us = make_windows(features, device_time_us)

    WINDOWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, windows=windows, window_start_time_us=starts_us)
    return out_path


def build_dataset(manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """All windows for every session in `manifest`, stacked into one [N, 400, 256] array, plus a
    per-window metadata DataFrame (session_dir, person_id, date) aligned to it by row index."""
    all_windows, meta_rows = [], []
    for _, row in manifest.iterrows():
        cache_path = build_session_windows(row["session_dir"])
        with np.load(cache_path) as d:
            windows, starts_us = d["windows"], d["window_start_time_us"]
        if len(windows) == 0:
            print(f"  skipping {row['session_dir']}: no windows (< {window_sec_str()} of usable data)")
            continue
        all_windows.append(windows)
        for s in starts_us:
            meta_rows.append({"session_dir": row["session_dir"], "person_id": row["person_id"],
                               "date": row["date"], "window_start_time_us": int(s)})
    windows_all = np.concatenate(all_windows, axis=0) if all_windows else np.empty((0, FINAL_LEN, 256), np.float32)
    meta = pd.DataFrame(meta_rows)
    print(f"built {len(meta)} windows ({WINDOW_SEC:.1f}s each, {OVERLAP:.0%} overlap) "
          f"from {len(manifest)} walking sessions")
    return windows_all, meta


def window_sec_str() -> str:
    return f"{WINDOW_SEC:.1f}s"


# --- per-fold reduction/normalization (fit on TRAIN windows only, applied frozen to TEST windows) ---

def select_topk_variance(windows: np.ndarray, k: int = TOP_K_SUBCARRIERS) -> np.ndarray:
    """Indices of the k highest-variance subcarriers, ranked by AMPLITUDE variance over
    (window, time) -- spec step 7. `windows` here must be TRAIN-fold windows only."""
    n_sub = windows.shape[2] // 2
    amp = windows[:, :, :n_sub]
    var = amp.var(axis=(0, 1))
    idx = np.argsort(var)[-k:]
    idx.sort()
    return idx


def apply_topk(windows: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """[N, 400, 256] (128 amp + 128 phase) -> [N, 400, 2k] (k amp + k phase), same idx for both halves."""
    n_sub = windows.shape[2] // 2
    amp_sel = windows[:, :, :n_sub][:, :, idx]
    phase_sel = windows[:, :, n_sub:][:, :, idx]
    return np.concatenate([amp_sel, phase_sel], axis=2)


def per_window_zscore(windows: np.ndarray) -> np.ndarray:
    """Per-window, per-feature-channel standardization over the time axis -- spec step 8, the
    cross-day AGC/scale-drift fix. Applied identically to train and test windows (each window
    normalizes against only its own statistics, so this never leaks across sessions/days)."""
    mu = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-6
    return ((windows - mu) / std).astype(np.float32)


if __name__ == "__main__":
    manifest = build_walk_manifest()
    print(f"walking channel-{TARGET_CHANNEL} sessions: {len(manifest)}")
    print(manifest.groupby(["date", "person_id"]).size())
    windows_all, meta = build_dataset(manifest)
    print(f"windows_all shape: {windows_all.shape}")
    print(meta.groupby(["date", "person_id"]).size())
