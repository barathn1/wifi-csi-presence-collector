"""Channel-6 preprocessing for the 2026-09-21 / 2026-09-22 multi-receiver (3 simultaneous ESP32s)
collection: auth (anjali, barath) vs. non_auth (every other person_id, plus empty-room `none`
sessions), for open-set identification, cross-day.

Same noise-reduction recipe this project already uses for its BiLSTM channel-6 model (see
`ablation_ch6_pipeline.py`'s E9 experiment and `paper_2507_12854_preprocessing.py`'s docstring):
IQ -> amplitude/phase -> phase unwrap -> Hampel filter (window=15, n_sigmas=3) -> Butterworth
low-pass (order=5, cutoff=10Hz) -> common-rate time resampling -> fixed 200-packet / 50%-overlap
windows. Only new here: sourcing from the multi-receiver session layout (`multi_receiver_session.py`,
one `<mac>_metadata.json`/`<mac>_samples.npz` pair per receiver per session dir) instead of the
older single-receiver one, and covering all three labels (authorized/unauthorized/none) rather than
authorized-only, since the open-set question needs real non-auth negatives.

Each of the 3 receivers' windows for a session are pooled as independent extra rows (no cross-
receiver fusion, no time-sync needed -- same approach this repo's other 2026-09-21+ scripts already
took): more observations of the same physical person/moment, not a new signal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT, channel_width_summary, decode_session_by_bucket
from ml.data_pipeline.multi_receiver_session import list_receivers, load_multi_receiver_session
from ml.data_pipeline.paper_2507_12854_preprocessing import butterworth_lowpass, hampel_filter
from ml.data_pipeline.time_resample import compute_target_rate_hz, resample_time_axis, session_native_rate_hz

TARGET_DATES = ["2026-09-21", "2026-09-22"]
TARGET_CHANNEL = 6
N_SUB_EXPECTED = 128
WINDOW_PACKETS = 200
STRIDE_PACKETS = 100
MAX_PACKETS_PER_SESSION = 100_000  # most sessions are ~2-5min (~50-95k packets); one 2026-09-22 `none`
# session is a 44-minute (~1.3M packet) outlier baseline capture that alone blew a Hampel+Butterworth
# denoised cache past 4GB per receiver -- capped here (after the monotonic-segment fix) rather than
# processed in full, since a single long empty-room capture adds nothing an already-adequate ~5min
# slice doesn't for the FAR-none diagnostic this project already treats as secondary to auth-vs-non-auth.
AUTH_PEOPLE = {"anjali", "barath"}
LABELS = ["authorized", "unauthorized", "none"]

DATA_DIR = REPO_ROOT / "data"
CACHE_ROOT = REPO_ROOT / "ml/data_pipeline/cache/sept2122_ch6_openset"
DENOISED_DIR = CACHE_ROOT / "denoised"
TIMENORM_DIR = CACHE_ROOT / "timenorm"


def build_manifest() -> pd.DataFrame:
    """One row per (session_dir, receiver_mac) for 2026-09-21/22, any label, channel-6 sessions only
    (validated per receiver, not assumed -- see module docstring). `is_auth` comes from `person_id`,
    not the folder label, since `person_id` is this project's documented authoritative signal for who
    a session actually is."""
    rows = []
    for label in LABELS:
        for date in TARGET_DATES:
            date_dir = DATA_DIR / label / date
            if not date_dir.exists():
                continue
            for session_dir in sorted(date_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                metas = sorted(session_dir.glob("*_metadata.json"))
                if not metas:
                    continue
                meta = json.loads(metas[0].read_text())
                person_id = meta.get("person_id") or ""
                for receiver_mac in list_receivers(session_dir):
                    rows.append({
                        "session_dir": str(session_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                        "receiver_mac": receiver_mac, "label": label, "person_id": person_id,
                        "is_auth": person_id in AUTH_PEOPLE, "motion": meta.get("motion") or "",
                        "date": date,
                    })
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest

    def _channel_primary(row):
        session = load_multi_receiver_session(REPO_ROOT / row["session_dir"], row["receiver_mac"])
        return channel_width_summary(session.npz)["channel_primary"]

    manifest["channel_primary"] = manifest.apply(_channel_primary, axis=1)
    dropped = manifest[manifest["channel_primary"] != TARGET_CHANNEL]
    if len(dropped):
        print(f"excluding {len(dropped)} non-channel-{TARGET_CHANNEL} (session,receiver) row(s): "
              f"{list(zip(dropped['session_dir'], dropped['receiver_mac']))}")
    manifest = manifest[manifest["channel_primary"] == TARGET_CHANNEL].drop(columns=["channel_primary"])
    return manifest.reset_index(drop=True)


def largest_monotonic_segment(device_time_us: np.ndarray) -> tuple[int, int]:
    """Some multi-receiver 2026-09-21+ recordings have a mid-session clock RESET in `device_time_us`
    (an isolated large backward jump, not a wraparound -- confirmed on this collection by earlier
    project work on the same 2026-09-21 receivers). Splits on every backward jump and keeps only the
    longest resulting contiguous non-decreasing run, so resampling never sees a corrupted time axis."""
    dt = np.diff(device_time_us)
    reset_idx = np.flatnonzero(dt < 0)
    if len(reset_idx) == 0:
        return 0, len(device_time_us)
    boundaries = [0] + [int(i) + 1 for i in reset_idx] + [len(device_time_us)]
    seg_lengths = np.diff(boundaries)
    best = int(np.argmax(seg_lengths))
    return boundaries[best], boundaries[best + 1]


def _safe_name(session_dir_rel: str, receiver_mac: str) -> str:
    return session_dir_rel.replace("/", "__") + "__" + receiver_mac.replace(":", "")


def decode_and_denoise(session_dir_rel: str, receiver_mac: str, force: bool = False) -> Path:
    """Decode this receiver's dominant channel-6 packet bucket, then: phase-unwrap -> Hampel
    (window=15, n_sigmas=3) -> Butterworth low-pass (order=5, cutoff=10Hz, fs=this session's own
    native packet rate). Amplitude and phase both get every stage (matches this project's existing
    bilstm_ch6 recipe), even though only amplitude is read by the classical features used downstream."""
    out_path = DENOISED_DIR / f"{_safe_name(session_dir_rel, receiver_mac)}.npz"
    if out_path.exists() and not force:
        return out_path

    session = load_multi_receiver_session(REPO_ROOT / session_dir_rel, receiver_mac)
    buckets = decode_session_by_bucket(session.npz)
    bucket = max(buckets.values(), key=lambda b: len(b["packet_indices"]))
    if bucket["n_subcarriers"] != N_SUB_EXPECTED:
        raise ValueError(f"{session_dir_rel} [{receiver_mac}]: dominant bucket has "
                          f"{bucket['n_subcarriers']} subcarriers, expected {N_SUB_EXPECTED}")

    device_time_us = session.npz["device_time_us"][bucket["packet_indices"]].astype(np.int64)
    rssi = session.npz["rssi"][bucket["packet_indices"]].astype(np.float32)
    amplitude, phase = bucket["amplitude"], bucket["phase"]

    seg_start, seg_end = largest_monotonic_segment(device_time_us)
    if seg_end - seg_start < len(device_time_us):
        print(f"  {session_dir_rel} [{receiver_mac}]: clock reset detected -- keeping largest "
              f"contiguous monotonic segment ({seg_end - seg_start}/{len(device_time_us)} packets)")
    device_time_us = device_time_us[seg_start:seg_end]
    amplitude, phase, rssi = amplitude[seg_start:seg_end], phase[seg_start:seg_end], rssi[seg_start:seg_end]

    if len(device_time_us) > MAX_PACKETS_PER_SESSION:
        print(f"  {session_dir_rel} [{receiver_mac}]: {len(device_time_us)} packets exceeds "
              f"{MAX_PACKETS_PER_SESSION} cap -- truncating to the first {MAX_PACKETS_PER_SESSION}")
        device_time_us = device_time_us[:MAX_PACKETS_PER_SESSION]
        amplitude, phase, rssi = (amplitude[:MAX_PACKETS_PER_SESSION], phase[:MAX_PACKETS_PER_SESSION],
                                   rssi[:MAX_PACKETS_PER_SESSION])

    native_rate_hz = session_native_rate_hz(device_time_us)
    phase = np.unwrap(phase.astype(np.float32), axis=0)

    amplitude = hampel_filter(amplitude.astype(np.float32), window=15, n_sigmas=3.0)
    phase = hampel_filter(phase, window=15, n_sigmas=3.0)
    amplitude = butterworth_lowpass(amplitude, fs=native_rate_hz, cutoff=10.0, order=5)
    phase = butterworth_lowpass(phase, fs=native_rate_hz, cutoff=10.0, order=5)

    DENOISED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, amplitude=amplitude.astype(np.float32), phase=phase.astype(np.float32),
              rssi=rssi, device_time_us=device_time_us, native_rate_hz=np.float64(native_rate_hz))
    return out_path


def compute_dataset_target_rate_hz(manifest: pd.DataFrame) -> float:
    rates = []
    for _, row in manifest.iterrows():
        with np.load(decode_and_denoise(row["session_dir"], row["receiver_mac"])) as d:
            rates.append(float(d["native_rate_hz"]))
    return compute_target_rate_hz(rates)


def time_normalize(session_dir_rel: str, receiver_mac: str, target_rate_hz: float, force: bool = False) -> Path:
    out_dir = TIMENORM_DIR / f"{target_rate_hz:.2f}hz"
    out_path = out_dir / f"{_safe_name(session_dir_rel, receiver_mac)}.npz"
    if out_path.exists() and not force:
        return out_path

    with np.load(decode_and_denoise(session_dir_rel, receiver_mac)) as d:
        amplitude, phase, rssi, device_time_us = d["amplitude"], d["phase"], d["rssi"], d["device_time_us"]
    amplitude, phase, rssi, device_time_us = resample_time_axis(amplitude, phase, device_time_us, rssi, target_rate_hz)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, amplitude=amplitude, phase=phase, rssi=rssi, device_time_us=device_time_us)
    return out_path


def build_window_index(manifest: pd.DataFrame, target_rate_hz: float,
                        window_packets: int = WINDOW_PACKETS, stride_packets: int = STRIDE_PACKETS) -> pd.DataFrame:
    """Schema matches `ml.data_pipeline.windowing`'s window index exactly (cache_path/start/end plus
    label metadata) so `features.build_feature_matrix` / `windowing.iter_windows` work unmodified."""
    rows = []
    for _, row in manifest.iterrows():
        cache_path = time_normalize(row["session_dir"], row["receiver_mac"], target_rate_hz)
        with np.load(cache_path, mmap_mode="r") as d:
            n = d["amplitude"].shape[0]
            device_time_us = d["device_time_us"]
            if n < window_packets:
                continue
            for start in range(0, n - window_packets + 1, stride_packets):
                end = start + window_packets
                rows.append({
                    "cache_path": str(cache_path), "start": start, "end": end,
                    "session_dir": row["session_dir"], "receiver_mac": row["receiver_mac"],
                    "label": row["label"], "person_id": row["person_id"], "is_auth": row["is_auth"],
                    "motion": row["motion"], "date": row["date"], "window_start_time_us": int(device_time_us[start]),
                })
    index = pd.DataFrame(rows)
    print(f"built {len(index)} windows ({window_packets} packets = {window_packets / target_rate_hz:.2f}s, "
          f"stride {stride_packets} packets, target_rate={target_rate_hz:.2f}Hz)")
    return index


def build_or_load_window_index(manifest: pd.DataFrame, target_rate_hz: float) -> pd.DataFrame:
    cache_csv = CACHE_ROOT / "window_index" / f"{target_rate_hz:.2f}hz_n{len(manifest)}.csv"
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    index = build_window_index(manifest, target_rate_hz)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(cache_csv, index=False)
    return index


if __name__ == "__main__":
    manifest = build_manifest()
    print(f"channel-6 (session,receiver) rows across {TARGET_DATES}: {len(manifest)}")
    print(manifest.groupby(["date", "label", "person_id"]).size())
