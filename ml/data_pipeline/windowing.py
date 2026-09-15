"""Turn decoded sessions into fixed-length CSI windows, without ever materializing the whole
overlapping-window dataset in RAM at once.

Every session mixes several `csi_len` (frame-type) buckets. Day 1 sessions were captured almost
entirely on a 40MHz-bonded channel (94.2-98.7% of packets at csi_len=372 bytes / 186 subcarriers), but
Day 2 sessions negotiated a 20MHz channel instead (~98-99% of packets at csi_len=256 bytes / 128
subcarriers) -- confirmed via the per-packet sig_mode/mcs/cwb/channel fields in samples.npz, not a
firmware change (config_snapshot is identical across days). So "the dominant bucket" is now detected
PER SESSION (`detect_dominant_csi_len`) rather than assumed to be one global constant. Two modes:
- `mode="native"`: cache each session at its own dominant subcarrier count. Correct, zero-distortion,
  but Day 1 (186 subcarriers) and Day 2 (128) windows are then different shapes -- fine for within-day
  analysis, but they can't share one model's weights for a real cross-day comparison.
- `mode="resampled"`: additionally resample every session's dominant bucket onto a fixed
  `csi_resample.TARGET_SUBCARRIERS`-point grid (128, the smaller of the two -- downsampling Day 1,
  never fabricating resolution Day 2 never captured). This is what day-disjoint splits and
  calibration Variant B need.
The minority buckets within a session (~1-6% of packets) are still dropped either way.

Observed packet rate on the dominant bucket is 162-263 Hz (mean ~217 Hz) across all 37 Day-1 sessions,
so window_packets=200 is close to a 1-second window (matches the ESP32 person-ID paper's convention);
window_packets=600 (~3s) is the second variant used in the combination sweep.

Memory design (this box runs with very little free RAM/swap alongside other sessions -- a first
version that stacked all ~15k overlapping windows into one array blew past ~9GB and had to be killed):
1. `cache_session()` decodes only the dominant bucket for one session ONCE and writes it to a small
   per-session .npz (amplitude/phase/rssi/device_time_us, float32). Largest session is ~160MB.
2. `build_window_index()` builds a pandas DataFrame of (cache_path, start, end, label, ...) -- just
   scalars, a few hundred KB total for the whole dataset, not the CSI data itself.
3. `load_window()` reads one window's slice from its per-session cache via `mmap_mode='r'`, so only
   that slice (~150KB) is ever pulled into RAM, on demand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.csi_resample import TARGET_SUBCARRIERS, resample_amplitude, resample_phase
from ml.data_pipeline.decode_csi import DATA_DIR, REPO_ROOT, decode_session_by_bucket, load_session

DOMINANT_CSI_LEN = 372  # bytes -> 186 subcarriers; Day-1-only legacy default, kept for old call sites
DOMINANT_CSI_LEN_20MHZ = 256  # bytes -> 128 subcarriers; Day 2's (and assumed Day 3's) dominant bucket
SESSION_CACHE_DIR = REPO_ROOT / "ml/data_pipeline/cache/sessions"


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "manifest.csv")


def detect_dominant_csi_len(npz: dict) -> int:
    """The csi_len value (bytes) that the most packets in this session share."""
    lens, counts = np.unique(npz["csi_len"], return_counts=True)
    return int(lens[np.argmax(counts)])


def _session_cache_path(session_dir_rel: str, mode: str, target_rate_hz: float | None = None) -> Path:
    safe_name = session_dir_rel.replace("/", "__")
    # target_rate_hz is folded into the filename (not just `mode`) so caches for two different derived
    # target rates -- e.g. re-deriving after adding faster-hardware sessions later -- never collide or
    # get silently reused from a stale, no-longer-correct rate.
    suffix = mode if target_rate_hz is None else f"{mode}_{target_rate_hz:.2f}hz"
    return SESSION_CACHE_DIR / f"{safe_name}__{suffix}.npz"


def cache_session(session_dir_rel: str, mode: str = "resampled", force: bool = False,
                   target_rate_hz: float | None = None) -> Path | None:
    """Decode one session's own dominant bucket once and cache it (auto-detected per session -- Day 1
    and Day 2 dominate on different frame formats, see module docstring). `mode`: "native" keeps the
    session's own subcarrier count; "resampled" additionally resamples onto the shared
    `TARGET_SUBCARRIERS`-point grid so sessions from either day are shape-compatible; "resampled_timenorm"
    additionally resamples the TIME axis onto `target_rate_hz` (required for this mode -- see
    time_resample.py::compute_target_rate_hz to derive it from the actual dataset in play, never a fixed
    constant) -- opt-in, used by ml/training/train_day3_ch6_model.py to close a real packet-rate/
    window-duration confound found in that collection (see [[project-day3-ch6-packet-rate-confound]]);
    every other existing caller is unaffected since this is a new, separately-cached mode value, not a
    change to "resampled"'s existing behavior. Returns None if the session has too few packets to have
    decoded at all."""
    assert mode in ("native", "resampled", "resampled_timenorm"), mode
    assert (mode == "resampled_timenorm") == (target_rate_hz is not None), \
        "target_rate_hz is required for (and only for) mode='resampled_timenorm'"
    out_path = _session_cache_path(session_dir_rel, mode, target_rate_hz)
    if out_path.exists() and not force:
        return out_path

    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    buckets = decode_session_by_bucket(session.npz)
    if not buckets:
        return None
    csi_len = detect_dominant_csi_len(session.npz)
    bucket = buckets[csi_len]

    amplitude, phase = bucket["amplitude"], bucket["phase"]
    if mode in ("resampled", "resampled_timenorm") and bucket["n_subcarriers"] != TARGET_SUBCARRIERS:
        amplitude = resample_amplitude(amplitude, TARGET_SUBCARRIERS)
        phase = resample_phase(phase, TARGET_SUBCARRIERS)

    packet_idx = bucket["packet_indices"]
    rssi = session.npz["rssi"][packet_idx].astype(np.float32)
    device_time_us = session.npz["device_time_us"][packet_idx].astype(np.int64)

    if mode == "resampled_timenorm":
        from ml.data_pipeline.time_resample import resample_time_axis
        amplitude, phase, rssi, device_time_us = resample_time_axis(
            amplitude, phase, device_time_us, rssi, target_rate_hz)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        amplitude=amplitude,
        phase=phase,
        rssi=rssi,
        device_time_us=device_time_us,
        source_csi_len=csi_len,
    )
    return out_path


def cache_all_sessions(manifest: pd.DataFrame | None = None, mode: str = "resampled") -> None:
    manifest = manifest if manifest is not None else load_manifest()
    for _, row in manifest.iterrows():
        path = cache_session(row["session_dir"], mode)
        n = "skipped (no packets)" if path is None else np.load(path)["amplitude"].shape[0]
        print(f"  {row['session_dir']}: {n} packets cached")


def build_window_index(
    manifest: pd.DataFrame | None = None,
    mode: str = "resampled",
    window_packets: int = 200,
    stride_packets: int | None = None,
    target_rate_hz: float | None = None,
) -> pd.DataFrame:
    """One row per window: where to find it (cache_path, start, end) + label metadata. No CSI data here.
    `target_rate_hz` is required for (and only for) mode="resampled_timenorm" -- see
    time_resample.py::compute_target_rate_hz."""
    manifest = manifest if manifest is not None else load_manifest()
    stride_packets = stride_packets or window_packets // 2

    rows = []
    for _, row in manifest.iterrows():
        cache_path = cache_session(row["session_dir"], mode, target_rate_hz=target_rate_hz)
        if cache_path is None:
            continue
        with np.load(cache_path, mmap_mode="r") as d:
            n = d["amplitude"].shape[0]
            device_time_us = d["device_time_us"]
            if n < window_packets:
                continue
            date = row["session_dir"].split("/")[1]
            for start in range(0, n - window_packets + 1, stride_packets):
                end = start + window_packets
                rows.append({
                    "cache_path": str(cache_path),
                    "start": start,
                    "end": end,
                    "session_dir": row["session_dir"],
                    "label": row["label"],
                    "person_id": "" if pd.isna(row["person_id"]) else row["person_id"],
                    "motion": "" if pd.isna(row["motion"]) else row["motion"],
                    "date": date,
                    "window_start_time_us": int(device_time_us[start]),
                })
    return pd.DataFrame(rows)


_SESSION_ARRAY_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _load_session_arrays(cache_path: str) -> dict[str, np.ndarray]:
    """Load one session's full decoded arrays into RAM once, then reuse across every window drawn from
    it. All 37 sessions' dominant-bucket arrays together are ~2.2GB (float32) -- fits comfortably, and
    avoids re-opening/re-parsing the .npz (a real bottleneck: reopening per window turned a few minutes
    of compute into 10+ minutes for the classical features, and would have been far worse across
    multiple training epochs for the deep models)."""
    cached = _SESSION_ARRAY_CACHE.get(cache_path)
    if cached is None:
        with np.load(cache_path) as d:
            cached = {"amplitude": d["amplitude"], "phase": d["phase"], "rssi": d["rssi"]}
        _SESSION_ARRAY_CACHE[cache_path] = cached
    return cached


def load_window(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one window's (amplitude, phase, rssi) slice from the in-memory session cache."""
    arrays = _load_session_arrays(row["cache_path"])
    sel = slice(int(row["start"]), int(row["end"]))
    return arrays["amplitude"][sel], arrays["phase"][sel], arrays["rssi"][sel]


def iter_windows(window_index: pd.DataFrame):
    """Yield (amplitude, phase, rssi, row) one window at a time -- for streaming feature extraction.

    Grouped by cache_path and kept open across consecutive windows from the same session: `load_window`
    reopening the per-session .npz on every single call (via np.load) dominated runtime once there were
    thousands of overlapping windows -- each open re-parses the zip's central directory.
    """
    for cache_path, group in window_index.groupby("cache_path", sort=False):
        with np.load(cache_path, mmap_mode="r") as d:
            amplitude, phase, rssi = d["amplitude"], d["phase"], d["rssi"]
            for _, row in group.iterrows():
                sel = slice(int(row["start"]), int(row["end"]))
                yield np.array(amplitude[sel]), np.array(phase[sel]), np.array(rssi[sel]), row


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window-packets", type=int, default=200)
    p.add_argument("--stride-packets", type=int, default=None)
    p.add_argument("--mode", choices=["native", "resampled"], default="resampled",
                   help="native = each session's own subcarrier count (days not shape-compatible); "
                        "resampled = every session resampled onto a shared subcarrier grid")
    args = p.parse_args()

    print(f"caching per-session dominant-bucket arrays (mode={args.mode})...")
    cache_all_sessions(mode=args.mode)

    stride = args.stride_packets or args.window_packets // 2
    index = build_window_index(mode=args.mode, window_packets=args.window_packets, stride_packets=stride)
    suffix = "" if args.mode == "resampled" else f"_{args.mode}"
    index_path = REPO_ROOT / "ml/data_pipeline/cache" / f"window_index_w{args.window_packets}_s{stride}{suffix}.csv"
    index.to_csv(index_path, index=False)
    print(f"built {len(index)} windows -> {index_path}")
    print(index["label"].value_counts())
    print(index.groupby("date")["label"].count())
