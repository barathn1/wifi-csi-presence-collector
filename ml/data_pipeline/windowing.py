"""Turn decoded sessions into fixed-length CSI windows, without ever materializing the whole
overlapping-window dataset in RAM at once.

Every session mixes several `csi_len` (frame-type) buckets, but csi_len=372 bytes (186 subcarriers,
the HT40 frame) dominates every single session in Day 1 (94.2-98.7% of packets per class, checked with
`python3 -m ml.data_pipeline.decode_csi <session> --buckets`). We filter to that one bucket so every
window has a consistent, fixed subcarrier count -- simplest correct choice for the first pass. The
minority buckets (~2-6% of packets) are dropped for now; padding+attention-masking to use them too is
listed as future work in the transformer models, not done here.

Observed packet rate on the dominant bucket is 162-263 Hz (mean ~217 Hz) across all 37 sessions, so
window_packets=200 is close to a 1-second window (matches the ESP32 person-ID paper's convention);
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

from ml.data_pipeline.decode_csi import DATA_DIR, REPO_ROOT, decode_session_by_bucket, load_session

DOMINANT_CSI_LEN = 372  # bytes -> 186 subcarriers; see module docstring
SESSION_CACHE_DIR = REPO_ROOT / "ml/data_pipeline/cache/sessions"


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "manifest.csv")


def _session_cache_path(session_dir_rel: str, csi_len: int) -> Path:
    safe_name = session_dir_rel.replace("/", "__")
    return SESSION_CACHE_DIR / f"{safe_name}__len{csi_len}.npz"


def cache_session(session_dir_rel: str, csi_len: int = DOMINANT_CSI_LEN, force: bool = False) -> Path | None:
    """Decode one session's dominant bucket once and cache it. Returns None if too few packets."""
    out_path = _session_cache_path(session_dir_rel, csi_len)
    if out_path.exists() and not force:
        return out_path

    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    buckets = decode_session_by_bucket(session.npz)
    bucket = buckets.get(csi_len)
    if bucket is None:
        return None

    packet_idx = bucket["packet_indices"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        amplitude=bucket["amplitude"],
        phase=bucket["phase"],
        rssi=session.npz["rssi"][packet_idx].astype(np.float32),
        device_time_us=session.npz["device_time_us"][packet_idx].astype(np.int64),
    )
    return out_path


def cache_all_sessions(manifest: pd.DataFrame | None = None, csi_len: int = DOMINANT_CSI_LEN) -> None:
    manifest = manifest if manifest is not None else load_manifest()
    for _, row in manifest.iterrows():
        path = cache_session(row["session_dir"], csi_len)
        n = "skipped (bucket absent)" if path is None else np.load(path)["amplitude"].shape[0]
        print(f"  {row['session_dir']}: {n} packets cached")


def build_window_index(
    manifest: pd.DataFrame | None = None,
    csi_len: int = DOMINANT_CSI_LEN,
    window_packets: int = 200,
    stride_packets: int | None = None,
) -> pd.DataFrame:
    """One row per window: where to find it (cache_path, start, end) + label metadata. No CSI data here."""
    manifest = manifest if manifest is not None else load_manifest()
    stride_packets = stride_packets or window_packets // 2

    rows = []
    for _, row in manifest.iterrows():
        cache_path = cache_session(row["session_dir"], csi_len)
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
    args = p.parse_args()

    print("caching per-session dominant-bucket arrays...")
    cache_all_sessions()

    stride = args.stride_packets or args.window_packets // 2
    index = build_window_index(window_packets=args.window_packets, stride_packets=stride)
    index_path = REPO_ROOT / "ml/data_pipeline/cache" / f"window_index_w{args.window_packets}_s{stride}.csv"
    index.to_csv(index_path, index=False)
    print(f"built {len(index)} windows -> {index_path}")
    print(index["label"].value_counts())
