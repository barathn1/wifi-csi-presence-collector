"""Consistent channel-6-only preprocessing for the BiLSTM person-ID task (2026-09-15/16/17,
anjali vs barath). Built from scratch, borrowing the one paper-grounded denoise step already
verified in this repo (`paper_2507_12854_preprocessing.preprocess_session`, the Avola et al.
Hampel+Butterworth+phase-calibration reconstruction) rather than re-deriving denoising from
scratch, then adding the two things a raw dual-branch LSTM needs that a hand-engineered-feature
model (the home model) doesn't: a common real-world packet rate (so "N packets" means the same
duration everywhere) and a common per-window scale (so absolute amplitude drift across sessions/
days doesn't dominate what the LSTM sees).

Per-session channel is read from the recorded `channel_primary` field, never assumed from the
date -- 2026-09-15's `20260915_143748_anjali` session is channel 11 (a mid-collection router hop),
not channel 6 like every other session across all three dates, and is silently wrong in this
repo's own prior docs (`ml/reports/home_model_and_evm_architecture.md` claims "Day3 = channel 6
for all"). `build_ch6_manifest` drops it by construction, not by a hardcoded exclusion list.

Pipeline per session, in order (matches the diagram exactly -- amplitude-only, no phase branch):
1. Validate packet metadata: drop `first_word_invalid` packets, drop duplicate-timestamp packets
   (keep first occurrence), then keep only packets addressed TO this session's own ESP32
   (`dst_mac == metadata.json's board_mac`) -- an evidence-based identification, not a guess: see
   `select_dominant_packets`'s docstring for the full src_mac/dst_mac pair analysis across all 39
   channel-6 sessions that established this is the one pair present in literally every session.
   `channel_primary` is still checked at the whole-session level too (`session_channel`/
   `build_ch6_manifest`) to drop the one channel-11 session entirely.
2. Within that identified link, keep only packets matching its own single DOMINANT (channel_primary,
   cwb, channel_secondary, stbc, mcs, csi_len) combination -- "keep one consistent CSI packet
   configuration." Every remaining packet shares one exact PHY config AND the one correct link.
3. Parse IQ pairs (`decode_session_by_bucket`, already validated against `inspect_npz.py`).
4. IQ -> amplitude (`np.hypot`).
5. Fixed 128 subcarriers, asserted (`N_SUB_EXPECTED`).
6. Outlier removal: Hampel filter (`paper_2507_12854_preprocessing.hampel_filter`).
7. Light smoothing: Butterworth low-pass (`...butterworth_lowpass`).
8. Normalization: per-window z-score (mean/std over the window's own time axis, per subcarrier),
   applied as a `CsiWindowDataset` calibration hook at __getitem__ time, NOT baked into the cache.
   Deliberately NOT empty-room baseline calibration (Variant A/B in calibration.py) -- this
   project's own signature-model experiments found empty-room calibration neutral-to-harmful for
   a different contrastive setup, and not every session here has a same-day `none` recording to
   calibrate against anyway. Per-window self-normalization needs no other session's data, so it
   can't leak anything across the day-disjoint splits used for cross-day evaluation.

One thing NOT dropped for cost reasons even though the diagram doesn't use it: phase is still
decoded, calibrated (Avola-paper endpoint-trend removal) and cached alongside amplitude, purely
because `preprocess_session`/`resample_time_axis` compute it as a side effect and other callers in
this repo expect the (amplitude, phase, rssi, device_time_us) cache schema. `window_zscore` and the
training script (`run_bilstm_ch6_day_to_day.py`) never read the phase array -- the model consumes
amplitude only, one [window_packets x 128] matrix per window, matching the diagram's BiLSTM input.

Also NOT in the diagram, kept anyway because this project already found it necessary once
(`train_day3_ch6_model.py`'s Day3-channel-6 packet-rate/window-duration confound): every session is
resampled onto one common real-world packet rate BEFORE windowing, so "200 packets" means the same
duration on every session/day. Skipping this would silently let the model learn packet-rate
(session/day) instead of identity, on data that's already known to vary 72-263Hz native rate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT, channel_width_summary, decode_session_by_bucket, load_session
from ml.data_pipeline.paper_2507_12854_preprocessing import preprocess_session
from ml.data_pipeline.time_resample import compute_target_rate_hz, resample_time_axis, session_native_rate_hz
from ml.data_pipeline.windowing import load_manifest

TARGET_DATES = ["2026-09-15", "2026-09-16", "2026-09-17"]
TARGET_CHANNEL = 6
N_SUB_EXPECTED = 128
WINDOW_PACKETS = 200   # literal packet count, matches the diagram's [200 x 128] input exactly
STRIDE_PACKETS = 100   # 50% overlap, this project's established default (windowing.py)

DENOISED_CACHE_DIR = REPO_ROOT / "ml/data_pipeline/cache/sessions_bilstm_ch6_denoised"
TIMENORM_CACHE_DIR = REPO_ROOT / "ml/data_pipeline/cache/sessions_bilstm_ch6_timenorm"


def session_channel(session_dir_rel: str) -> int:
    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    return channel_width_summary(session.npz)["channel_primary"]


def build_ch6_manifest() -> pd.DataFrame:
    """taskB (authorized anjali/barath) rows for 2026-09-15/16/17, channel-6 sessions only."""
    manifest = load_manifest()
    date_mask = manifest["session_dir"].apply(lambda s: any(f"/{d}/" in s for d in TARGET_DATES))
    auth = manifest[date_mask & (manifest["label"] == "authorized")].copy()
    auth["channel_primary"] = auth["session_dir"].apply(session_channel)
    dropped = auth[auth["channel_primary"] != TARGET_CHANNEL]
    if len(dropped):
        print(f"excluding {len(dropped)} non-channel-{TARGET_CHANNEL} session(s): "
              f"{dropped['session_dir'].tolist()} (channel={dropped['channel_primary'].tolist()})")
    ch6 = auth[auth["channel_primary"] == TARGET_CHANNEL].drop(columns=["channel_primary"])
    ch6["date"] = ch6["session_dir"].apply(lambda s: s.split("/")[1])
    return ch6.reset_index(drop=True)


def _reduce_1d(x: np.ndarray) -> np.ndarray:
    """Pairwise mean, matching paper_2507_12854_preprocessing.temporal_mean_reduce's halving of the
    time axis, for the 1D companion arrays (device_time_us, rssi) that function doesn't touch."""
    n = x.shape[0] - (x.shape[0] % 2)
    return ((x[:n:2].astype(np.float64) + x[1:n:2].astype(np.float64)) / 2.0)


# Fields that must all match, packet-for-packet, for a packet to count as "one consistent CSI
# packet configuration" -- diagram steps 1+2. csi_len is included because it's the direct
# consequence of (cwb, sig_mode/HT-mode) and is what actually determines the decoded shape.
_COMBO_FIELDS = ["channel_primary", "cwb", "channel_secondary", "stbc", "mcs", "csi_len"]


def mac_str_to_bytes(mac_str: str) -> np.ndarray:
    return np.array([int(b, 16) for b in mac_str.split(":")], dtype=np.uint8)


def select_dominant_packets(npz: dict, board_mac: str) -> np.ndarray:
    """Validate packet metadata and return the indices of the intended CSI transmitter-to-ESP32
    link's packets: not first_word_invalid, not a duplicate timestamp, addressed TO this session's
    own ESP32 (`dst_mac == board_mac`, from metadata.json -- never hardcoded, never "whichever MAC
    pair happens to be most common"), then further restricted to that link's own single dominant
    (channel/cwb/stbc/mcs/csi_len) combo.

    The `dst_mac == board_mac` identification is evidence-based, not assumed: an explicit analysis
    of every distinct (src_mac, dst_mac) pair across all 39 channel-6 sessions (2026-09-15/16/17)
    found exactly 4 pairs total. `(d8:47:32:79:d4:68 -> board_mac)` is the ONLY pair present in
    every single session (80.97-99.90% of that session's packets each, 97.0% pooled) -- this is the
    router transmitting directly to the ESP32, the actual sensing link. The other 3 pairs are
    either the reverse/uplink half of the same stimulus conversation (laptop -> router, overheard
    promiscuously -- a DIFFERENT physical transmitter position, so a different propagation path,
    not more data about the same link) or one session's 274 packets (0.02% overall) from an
    unrelated locally-administered/randomized MAC device. Filtering on the most-common pair alone
    (this pipeline's previous approach) happened to agree with this identification on every session
    here, but only because the intended link was always the majority one; filtering by board_mac
    identity is the only version of this that keeps being correct if a session's dominant traffic
    were ever NOT the intended link (e.g. heavy interference from another device)."""
    valid = ~npz["first_word_invalid"].astype(bool)

    device_time_us = npz["device_time_us"]
    _, first_occurrence = np.unique(device_time_us, return_index=True)
    keep_first = np.zeros(len(device_time_us), dtype=bool)
    keep_first[first_occurrence] = True
    valid &= keep_first

    board_mac_bytes = mac_str_to_bytes(board_mac)
    valid &= np.all(npz["dst_mac"] == board_mac_bytes, axis=1)

    valid_idx = np.flatnonzero(valid)
    combo = np.stack([npz[f][valid_idx] for f in _COMBO_FIELDS], axis=1)
    uniq_combo, combo_counts = np.unique(combo, axis=0, return_counts=True)
    dominant_combo = uniq_combo[np.argmax(combo_counts)]
    combo_mask = np.all(combo == dominant_combo, axis=1)
    return valid_idx[combo_mask]


def denoise_session(session_dir_rel: str, force: bool = False) -> Path:
    """Validate packet metadata, decode, then Hampel/Butterworth/phase-calibrate one session,
    caching the halved-rate result."""
    safe_name = session_dir_rel.replace("/", "__")
    out_path = DENOISED_CACHE_DIR / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    dominant_idx = select_dominant_packets(session.npz, session.metadata["board_mac"])
    dropped = len(session.npz["csi_len"]) - len(dominant_idx)
    if dropped:
        print(f"  {session_dir_rel}: dropped {dropped} packet(s) "
              f"({100 * dropped / len(session.npz['csi_len']):.2f}%) failing metadata validation "
              f"(wrong link and/or inconsistent PHY config)")

    sub_npz = {k: v[dominant_idx] for k, v in session.npz.items() if k != "csi_flat"}
    sub_npz["csi_flat"] = session.npz["csi_flat"]  # unchanged -- csi_offset (just re-selected) still indexes into it
    buckets = decode_session_by_bucket(sub_npz)
    assert len(buckets) == 1, f"{session_dir_rel}: metadata validation should leave exactly one csi_len bucket"
    bucket = next(iter(buckets.values()))
    if bucket["n_subcarriers"] != N_SUB_EXPECTED:
        raise ValueError(f"{session_dir_rel}: validated packets have {bucket['n_subcarriers']} "
                          f"subcarriers, expected {N_SUB_EXPECTED}")

    device_time_us = sub_npz["device_time_us"].astype(np.int64)
    rssi = sub_npz["rssi"].astype(np.float32)
    native_rate_hz = session_native_rate_hz(device_time_us)

    amp, phase, effective_rate_hz = preprocess_session(bucket["amplitude"], bucket["phase"], native_rate_hz)
    device_time_us = _reduce_1d(device_time_us).astype(np.int64)
    rssi = _reduce_1d(rssi).astype(np.float32)
    assert len(device_time_us) == len(amp), (len(device_time_us), len(amp))

    DENOISED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, amplitude=amp, phase=phase, rssi=rssi, device_time_us=device_time_us,
              effective_rate_hz=np.float64(effective_rate_hz))
    return out_path


def compute_dataset_target_rate_hz(manifest: pd.DataFrame) -> float:
    rates = []
    for session_dir in manifest["session_dir"]:
        with np.load(denoise_session(session_dir)) as d:
            rates.append(float(d["effective_rate_hz"]))
    target = compute_target_rate_hz(rates)
    print(f"post-denoise effective rates: {min(rates):.1f}-{max(rates):.1f} Hz across {len(rates)} "
          f"channel-{TARGET_CHANNEL} sessions -> common target: {target:.2f} Hz")
    return target


def time_normalize_session(session_dir_rel: str, target_rate_hz: float, force: bool = False) -> Path:
    safe_name = session_dir_rel.replace("/", "__")
    out_dir = TIMENORM_CACHE_DIR / f"{target_rate_hz:.2f}hz"
    out_path = out_dir / f"{safe_name}.npz"
    if out_path.exists() and not force:
        return out_path

    with np.load(denoise_session(session_dir_rel)) as d:
        amp, phase, rssi, device_time_us = d["amplitude"], d["phase"], d["rssi"], d["device_time_us"]
    amp_r, phase_r, rssi_r, device_time_us_r = resample_time_axis(amp, phase, device_time_us, rssi, target_rate_hz)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, amplitude=amp_r, phase=phase_r, rssi=rssi_r, device_time_us=device_time_us_r)
    return out_path


def build_window_index(manifest: pd.DataFrame, target_rate_hz: float,
                        window_packets: int = WINDOW_PACKETS, stride_packets: int = STRIDE_PACKETS) -> pd.DataFrame:
    """One row per fixed-length window -- same schema windowing.py's build_window_index produces,
    so CsiWindowDataset/load_window work unmodified. window_packets is a literal packet count
    (matches the diagram's [200 x 128] exactly); because every session was already resampled onto
    the same target_rate_hz, 200 packets is also the same real-world duration everywhere."""
    rows = []
    for _, row in manifest.iterrows():
        cache_path = time_normalize_session(row["session_dir"], target_rate_hz)
        with np.load(cache_path, mmap_mode="r") as d:
            n = d["amplitude"].shape[0]
            device_time_us = d["device_time_us"]
            if n < window_packets:
                print(f"  skipping {row['session_dir']}: only {n} packets < window size {window_packets}")
                continue
            for start in range(0, n - window_packets + 1, stride_packets):
                end = start + window_packets
                rows.append({
                    "cache_path": str(cache_path), "start": start, "end": end,
                    "session_dir": row["session_dir"], "label": row["label"],
                    "person_id": row["person_id"], "motion": row.get("motion", ""),
                    "date": row["date"], "window_start_time_us": int(device_time_us[start]),
                })
    index = pd.DataFrame(rows)
    print(f"built {len(index)} windows ({window_packets} packets = {window_packets / target_rate_hz:.2f}s, "
          f"stride {stride_packets} packets)")
    return index


def window_zscore(amp: np.ndarray, phase: np.ndarray, row) -> tuple[np.ndarray, np.ndarray]:
    """Per-window, per-subcarrier standardization -- the 'consistent scale' step. No other
    session's statistics are used, so this can never leak across a day-disjoint split."""
    eps = 1e-6
    amp_z = (amp - amp.mean(axis=0, keepdims=True)) / (amp.std(axis=0, keepdims=True) + eps)
    phase_z = (phase - phase.mean(axis=0, keepdims=True)) / (phase.std(axis=0, keepdims=True) + eps)
    return amp_z.astype(np.float32), phase_z.astype(np.float32)


def build_or_load_window_index(cache_csv: Path, manifest: pd.DataFrame, target_rate_hz: float) -> pd.DataFrame:
    if cache_csv.exists():
        return pd.read_csv(cache_csv)
    index = build_window_index(manifest, target_rate_hz)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(cache_csv, index=False)
    return index


if __name__ == "__main__":
    manifest = build_ch6_manifest()
    print(f"channel-{TARGET_CHANNEL} sessions: {len(manifest)}")
    print(manifest.groupby(["date", "person_id"]).size())
    target_rate_hz = compute_dataset_target_rate_hz(manifest)
    index = build_window_index(manifest, target_rate_hz)
    print(index.groupby(["date", "person_id"]).size())
