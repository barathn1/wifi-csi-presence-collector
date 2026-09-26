"""Decode raw ESP32 CSI packets from a session's samples.npz into amplitude/phase arrays.

Independent, from-scratch implementation -- does not import anything from
wifi-csi-presence-collector's ml/ code. The wire format itself is a hardware/firmware fact, verified
here empirically rather than taken from that repo's source:

  samples.npz['csi_flat'] is one flat int8 buffer for the whole session. Packet i's CSI lives at
  csi_flat[csi_offset[i] : csi_offset[i] + csi_len[i]], as (imag, real) int8 byte pairs, one pair per
  subcarrier (so csi_len[i] // 2 subcarriers). Not every packet in a session has the same csi_len
  (different 802.11 frame types carry different subcarrier counts), so packets are bucketed by csi_len
  and only the dominant bucket is kept.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz(path: Path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def dominant_csi_len(npz: dict) -> int:
    lens, counts = np.unique(npz["csi_len"], return_counts=True)
    return int(lens[np.argmax(counts)])


def decode_dominant_bucket(npz: dict) -> dict:
    """Amplitude/phase/rssi/device_time_us/channel info for just the dominant csi_len bucket."""
    csi_len = npz["csi_len"]
    length = dominant_csi_len(npz)
    idx = np.flatnonzero(csi_len == length)

    csi_offset = npz["csi_offset"].astype(np.int64)[idx]
    n_sub = length // 2
    gather = csi_offset[:, None] + np.arange(length)[None, :]
    raw = npz["csi_flat"][gather].astype(np.float32)  # (n_packets, length): [imag0,real0,imag1,real1,...]
    imag = raw[:, 0::2]
    real = raw[:, 1::2]

    return {
        "amplitude": np.hypot(real, imag).astype(np.float32),   # (n_packets, n_sub)
        "phase": np.arctan2(imag, real).astype(np.float32),
        "rssi": npz["rssi"][idx].astype(np.float32),
        "device_time_us": npz["device_time_us"][idx].astype(np.int64),
        "n_subcarriers": n_sub,
        "csi_len": length,
        "packet_indices": idx,
    }


def decode_one_packet_bytes(csi_data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode ONE live packet's raw CSI bytes (same (imag,real) int8-pair layout as the batched
    decoder above, just for a single already-arrived packet instead of a whole recorded session's
    array) -- used by the live bridge, where packets arrive one at a time rather than as one big
    array already in memory."""
    raw = np.frombuffer(csi_data, dtype=np.int8).astype(np.float32)
    imag, real = raw[0::2], raw[1::2]
    return np.hypot(real, imag).astype(np.float32), np.arctan2(imag, real).astype(np.float32)


def monotonic_elapsed_seconds(device_time_us: np.ndarray) -> np.ndarray:
    """Cumulative elapsed seconds from device_time_us, guarded against the known ESP32 clock-reset
    quirk (the counter occasionally jumps backward mid-session -- seen in ~6 of 44 sessions in this
    dataset, e.g. a -6703s "elapsed time" if used naively). A backward jump is treated as zero elapsed
    time for that one step rather than a huge negative delta -- doesn't recover the lost timing
    precision around the reset, but keeps a live elapsed-time display from breaking (going negative or
    exploding), which is what actually matters for something like live_inference.py's reveal-timer."""
    diffs = np.diff(device_time_us.astype(np.int64))
    diffs = np.clip(diffs, 0, None)
    cumulative = np.concatenate([[0], np.cumsum(diffs)])
    return cumulative / 1e6


def dominant_channel(npz: dict) -> tuple[int, int]:
    """(channel_primary, cwb) of whichever combo most packets share -- cwb 0=20MHz, 1=40MHz."""
    combos, counts = np.unique(
        np.stack([npz["channel_primary"], npz["cwb"]], axis=1), axis=0, return_counts=True)
    return tuple(int(v) for v in combos[np.argmax(counts)])
