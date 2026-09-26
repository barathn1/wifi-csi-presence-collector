"""2026-09-21 multi-receiver walking data (anjali, barath, plus "stranger" candidates promoda/
sumanth), preprocessed the same way as `walk_bilstm_pipeline.py` (same Hampel/Butterworth/phase-
sanitize/time-windowing constants, same `sanitize_phase`/`select_topk_variance`/`apply_topk`/
`per_window_zscore` functions, imported unchanged) but sourced from the new multi-receiver session
layout (`multi_receiver_session.py`) instead of the older single-receiver one.

Two things this collection changed that the manifest-building here has to account for, found by
inspecting the actual files rather than assumed from the folder layout:
1. Every 2026-09-21 session -- anjali, barath, AND promoda/sumanth alike -- is filed under the
   `authorized` label folder with `metadata["label"] == "authorized"`. There is no `unauthorized/`
   folder for this date at all. `person_id` is the only reliable signal for "who is this" -- this
   module treats anjali/barath as the enrolled identities and everyone else's person_id as an
   unauthorized/stranger candidate for open-set purposes, ignoring the metadata `label` field for this
   date entirely (it does not mean what it means for every other date in this repo).
2. Not every receiver captured a usable `samples.npz` for every session -- `list_receivers` reports
   only the receivers that actually did. Several sessions (both of promoda's, both of sumanth's, as
   of this writing) have ZERO usable receivers -- likely files not yet synced/downloaded rather than
   a real capture failure, but this module treats "zero usable receivers" as "skip this session" no
   matter the reason, rather than guessing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.bilstm_ch6_pipeline import N_SUB_EXPECTED, TARGET_CHANNEL, select_dominant_packets
from ml.data_pipeline.decode_csi import DATA_DIR, REPO_ROOT, channel_width_summary, decode_session_by_bucket
from ml.data_pipeline.multi_receiver_session import list_receivers, load_multi_receiver_session
from ml.data_pipeline.paper_2507_12854_preprocessing import butterworth_lowpass, hampel_filter
from ml.data_pipeline.time_resample import session_native_rate_hz
from ml.data_pipeline.walk_bilstm_pipeline import (
    BUTTER_CUTOFF_HZ,
    BUTTER_ORDER,
    HAMPEL_N_SIGMAS,
    HAMPEL_WINDOW,
    make_windows,
    sanitize_phase,
)

SEPT21_DATE = "2026-09-21"
ENROLLED_PEOPLE = {"anjali", "barath"}
CACHE_ROOT = REPO_ROOT / "ml/data_pipeline/cache/walk_bilstm_sept21"
DECODED_CACHE_DIR = CACHE_ROOT / "decoded"
WINDOWS_CACHE_DIR = CACHE_ROOT / "windows"


def build_sept21_manifest() -> pd.DataFrame:
    """One row per (session, receiver) with a usable npz, walking-only, any person_id (enrolled or
    stranger candidate) -- `person_id` is authoritative here, NOT the metadata `label` field (see
    module docstring: every 2026-09-21 session is filed as "authorized" regardless of who it is)."""
    base = DATA_DIR / "authorized" / SEPT21_DATE
    rows = []
    if not base.exists():
        return pd.DataFrame(rows)
    for session_dir in sorted(base.iterdir()):
        if not session_dir.is_dir():
            continue
        # person_id/motion are identical across every receiver's metadata.json for one session
        # (same physical event, multiple simultaneous observers) -- read from whichever exists.
        any_meta = next(session_dir.glob("*_metadata.json"), None)
        if any_meta is None:
            continue
        import json
        meta = json.loads(any_meta.read_text())
        if meta.get("motion") != "walking":
            continue
        for receiver_mac in list_receivers(session_dir):
            rows.append({
                "session_dir": str(session_dir.relative_to(REPO_ROOT)), "receiver_mac": receiver_mac,
                "person_id": meta.get("person_id"), "is_enrolled": meta.get("person_id") in ENROLLED_PEOPLE,
                "date": SEPT21_DATE,
            })
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest
    manifest["channel_primary"] = manifest.apply(
        lambda r: channel_width_summary(
            load_multi_receiver_session(REPO_ROOT / r["session_dir"], r["receiver_mac"]).npz
        )["channel_primary"], axis=1)
    dropped = manifest[manifest["channel_primary"] != TARGET_CHANNEL]
    if len(dropped):
        print(f"excluding {len(dropped)} non-channel-{TARGET_CHANNEL} (session,receiver) row(s)")
    return manifest[manifest["channel_primary"] == TARGET_CHANNEL].drop(columns=["channel_primary"]).reset_index(drop=True)


def largest_monotonic_segment(device_time_us: np.ndarray) -> tuple[int, int]:
    """Several 2026-09-21 (session,receiver) recordings have a mid-session clock RESET -- one or two
    isolated, huge (hundreds of millions of microseconds) backward jumps in `device_time_us`, after
    which normal small increments resume from the new, lower reference (not a rejoined/self-correcting
    glitch, and not a clean 32-bit-counter wraparound either -- the whole-session start-to-end delta
    stays negative, confirmed by inspecting several affected sessions directly). Splits on every
    backward jump and returns the [start, end) slice of the longest resulting contiguous
    non-decreasing run, so the rest of this module never sees a corrupted time axis."""
    dt = np.diff(device_time_us)
    reset_idx = np.flatnonzero(dt < 0)
    if len(reset_idx) == 0:
        return 0, len(device_time_us)
    boundaries = [0] + [int(i) + 1 for i in reset_idx] + [len(device_time_us)]
    seg_lengths = np.diff(boundaries)
    best = int(np.argmax(seg_lengths))
    return boundaries[best], boundaries[best + 1]


def _safe_name(session_dir_rel: str, receiver_mac: str) -> str:
    normalized = session_dir_rel.replace("\\", "/").replace("/", "__")
    return normalized + "__" + receiver_mac.replace(":", "")


def decode_and_denoise(session_dir_rel: str, receiver_mac: str, force: bool = False) -> Path:
    """Validate (this receiver's own dominant PHY combo + dst_mac==its own board_mac) -> decode ->
    phase-sanitize -> Hampel -> Butterworth. Same steps/constants as
    `walk_bilstm_pipeline.denoise_session`, adapted only for the multi-receiver session source."""
    out_path = DECODED_CACHE_DIR / f"{_safe_name(session_dir_rel, receiver_mac)}.npz"
    if out_path.exists() and not force:
        return out_path

    session = load_multi_receiver_session(REPO_ROOT / session_dir_rel, receiver_mac)
    dominant_idx = select_dominant_packets(session.npz, session.metadata["board_mac"])
    sub_npz = {k: v[dominant_idx] for k, v in session.npz.items() if k != "csi_flat"}
    sub_npz["csi_flat"] = session.npz["csi_flat"]
    buckets = decode_session_by_bucket(sub_npz)
    bucket = max(buckets.values(), key=lambda b: len(b["packet_indices"]))
    if bucket["n_subcarriers"] != N_SUB_EXPECTED:
        raise ValueError(f"{session_dir_rel} [{receiver_mac}]: {bucket['n_subcarriers']} subcarriers, "
                          f"expected {N_SUB_EXPECTED}")

    device_time_us = sub_npz["device_time_us"][bucket["packet_indices"]].astype(np.int64)
    amplitude, phase = bucket["amplitude"], bucket["phase"]

    seg_start, seg_end = largest_monotonic_segment(device_time_us)
    if seg_end - seg_start < len(device_time_us):
        print(f"  {session_dir_rel} [{receiver_mac}]: mid-session clock reset detected "
              f"({len(device_time_us)} packets total) -- keeping largest contiguous monotonic "
              f"segment only ({seg_end - seg_start} packets, [{seg_start}:{seg_end}])")
    device_time_us = device_time_us[seg_start:seg_end]
    amplitude, phase = amplitude[seg_start:seg_end], phase[seg_start:seg_end]
    native_rate_hz = session_native_rate_hz(device_time_us)

    phase_sanitized = sanitize_phase(phase.astype(np.float32))
    features = np.concatenate([amplitude.astype(np.float32), phase_sanitized], axis=1)
    features = hampel_filter(features, window=HAMPEL_WINDOW, n_sigmas=HAMPEL_N_SIGMAS)
    features = butterworth_lowpass(features, fs=native_rate_hz, cutoff=BUTTER_CUTOFF_HZ, order=BUTTER_ORDER)

    DECODED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, features=features.astype(np.float32), device_time_us=device_time_us,
              native_rate_hz=np.float64(native_rate_hz))
    return out_path


def build_session_windows(session_dir_rel: str, receiver_mac: str, force: bool = False) -> Path:
    out_path = WINDOWS_CACHE_DIR / f"{_safe_name(session_dir_rel, receiver_mac)}.npz"
    if out_path.exists() and not force:
        return out_path
    with np.load(decode_and_denoise(session_dir_rel, receiver_mac)) as d:
        features, device_time_us = d["features"], d["device_time_us"]
    windows, starts_us = make_windows(features, device_time_us)
    WINDOWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, windows=windows, window_start_time_us=starts_us)
    return out_path


def build_dataset(manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    all_windows, meta_rows = [], []
    for _, row in manifest.iterrows():
        cache_path = build_session_windows(row["session_dir"], row["receiver_mac"])
        with np.load(cache_path) as d:
            windows, starts_us = d["windows"], d["window_start_time_us"]
        if len(windows) == 0:
            print(f"  skipping {row['session_dir']} [{row['receiver_mac']}]: no windows")
            continue
        all_windows.append(windows)
        for s in starts_us:
            meta_rows.append({"session_dir": row["session_dir"], "receiver_mac": row["receiver_mac"],
                               "person_id": row["person_id"], "is_enrolled": row["is_enrolled"],
                               "date": row["date"], "window_start_time_us": int(s)})
    windows_all = np.concatenate(all_windows, axis=0) if all_windows else np.empty((0, 400, 256), np.float32)
    meta = pd.DataFrame(meta_rows)
    print(f"built {len(meta)} windows from {len(manifest)} (session,receiver) rows")
    return windows_all, meta


if __name__ == "__main__":
    manifest = build_sept21_manifest()
    print(f"2026-09-21 walking (session,receiver) rows: {len(manifest)}")
    if not manifest.empty:
        print(manifest.groupby(["person_id", "receiver_mac"]).size())
