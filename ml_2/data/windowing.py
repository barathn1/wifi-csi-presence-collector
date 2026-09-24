"""Decode + cache each (session, receiver) unit once, then slice fixed-size, overlapping windows from
the cache. Simplification versus a full time-normalization scheme: windows are fixed PACKET counts
(not seconds) -- native packet rate varies session to session (roughly 130-600 Hz observed across this
dataset), so a 200-packet window is ~0.3-1.5s depending on the session, not a constant duration. This
is a disclosed simplification, not a hidden one -- see ml_2/PLAN.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_2.data.decode import DATA_DIR, decode_session_amplitude_phase

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "sessions"

# Channel-6 sessions are cwb=0 (20MHz HT20) but the firmware still merges LLTF+HT-LTF into a 128-value
# buffer (lltf_en=htltf_en=ltf_merge_en=true in every session's config_snapshot) -- confirmed directly
# from the raw per-packet fields, not assumed. Keeping only the first 64 columns -- the LEGACY (LLTF)
# subcarriers, present in every 802.11 frame type and the more standard/comparable set in the CSI
# literature -- rather than the full 128 (which includes the supplementary HT-LTF half).
N_LEGACY_SUBCARRIERS = 64


def _cache_path(session_dir: str, receiver_mac: str) -> Path:
    safe = session_dir.replace("/", "__")
    return CACHE_DIR / f"{safe}__{receiver_mac.replace(':', '')}.npz"


def cache_row(row: pd.Series, force: bool = False) -> Path | None:
    out_path = _cache_path(row["session_dir"], row["receiver_mac"])
    if out_path.exists() and not force:
        return out_path
    board_mac = row["receiver_mac"] if row.get("is_prefixed", False) else None
    decoded = decode_session_amplitude_phase(DATA_DIR / row["session_dir"], board_mac=board_mac)
    if decoded is None:
        return None
    amplitude = decoded["amplitude"][:, :N_LEGACY_SUBCARRIERS]
    phase = decoded["phase"][:, :N_LEGACY_SUBCARRIERS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, amplitude=amplitude, phase=phase,
             rssi=decoded["rssi"] if decoded["rssi"] is not None else np.zeros(len(amplitude), np.float32))
    return out_path


def cache_all(manifest: pd.DataFrame) -> pd.DataFrame:
    """Adds a `cache_path` column, dropping rows that failed to decode."""
    manifest = manifest.copy()
    paths = []
    for _, row in manifest.iterrows():
        paths.append(cache_row(row))
    manifest["cache_path"] = paths
    n_before = len(manifest)
    manifest = manifest[manifest["cache_path"].notna()].reset_index(drop=True)
    if len(manifest) < n_before:
        print(f"  dropped {n_before - len(manifest)} rows that failed to decode")
    return manifest


def build_window_index(manifest: pd.DataFrame, window_packets: int = 200, stride_packets: int | None = None,
                        min_packets: int = 50) -> pd.DataFrame:
    stride_packets = stride_packets or window_packets // 2
    rows = []
    for _, row in manifest.iterrows():
        with np.load(row["cache_path"], mmap_mode="r") as d:
            n = d["amplitude"].shape[0]
        if n < window_packets:
            continue
        for start in range(0, n - window_packets + 1, stride_packets):
            rows.append({
                "cache_path": str(row["cache_path"]), "start": start, "end": start + window_packets,
                "session_dir": row["session_dir"], "receiver_mac": row["receiver_mac"],
                "label": row["label"], "person_id": row["person_id"], "date": row["date"],
            })
    index = pd.DataFrame(rows)
    print(f"window index: {len(index)} windows from {index['session_dir'].nunique() if len(index) else 0} sessions")
    return index


_ARRAY_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _arrays(cache_path: str) -> dict[str, np.ndarray]:
    cached = _ARRAY_CACHE.get(cache_path)
    if cached is None:
        with np.load(cache_path) as d:
            cached = {"amplitude": d["amplitude"], "phase": d["phase"], "rssi": d["rssi"]}
        _ARRAY_CACHE[cache_path] = cached
    return cached


def load_window(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = _arrays(row["cache_path"])
    sel = slice(int(row["start"]), int(row["end"]))
    return arrays["amplitude"][sel], arrays["phase"][sel], arrays["rssi"][sel]


def iter_windows(window_index: pd.DataFrame):
    for cache_path, group in window_index.groupby("cache_path", sort=False):
        with np.load(cache_path, mmap_mode="r") as d:
            amplitude, phase, rssi = d["amplitude"], d["phase"], d["rssi"]
            for _, row in group.iterrows():
                sel = slice(int(row["start"]), int(row["end"]))
                yield np.array(amplitude[sel]), np.array(phase[sel]), np.array(rssi[sel]), row
