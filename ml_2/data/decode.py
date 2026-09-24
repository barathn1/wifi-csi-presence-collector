"""Self-contained CSI decode, independent of ml/data_pipeline. `samples.npz` stores CSI as a flat,
ragged buffer: `csi_flat[csi_offset[i] : csi_offset[i]+csi_len[i]]` holds `csi_len[i] // 2` (imag,real)
int8 pairs (one pair per subcarrier) for packet i. `csi_len` varies within a session (different
802.11 frame types/bandwidths carry different subcarrier counts), so packets are bucketed by csi_len
and each bucket decoded in one vectorized shot.

Channel detection: the WiFi channel actually negotiated over the air is `channel_primary`, a per-packet
field the firmware records from its own radio config -- NOT `metadata.json`'s
`config_snapshot.hotspot.channel`, which is always `1` on every session in this dataset regardless of
the real channel (it's a serial/hotspot config value, unrelated). This is re-derived here directly from
the raw per-packet field, not assumed from any prior finding.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class RawSession(NamedTuple):
    session_dir: Path
    board_mac: str | None  # None for single-receiver sessions (unprefixed samples.npz)
    metadata: dict
    npz: dict


def _mac_no_colons(mac: str) -> str:
    return mac.replace(":", "").lower()


def list_receiver_macs(session_dir: Path) -> list[str | None]:
    """[None] for a single-receiver session; otherwise every board MAC that actually wrote a
    `<mac>_samples.npz` file in this session directory (multi-receiver sessions)."""
    if (session_dir / "samples.npz").exists():
        return [None]
    macs = []
    for meta_path in sorted(session_dir.glob("*_metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        mac = meta.get("board_mac")
        if mac and (session_dir / f"{_mac_no_colons(mac)}_samples.npz").exists():
            macs.append(mac)
    return macs


def load_raw_session(session_dir: Path, board_mac: str | None = None) -> RawSession:
    prefix = f"{_mac_no_colons(board_mac)}_" if board_mac else ""
    meta_path = session_dir / f"{prefix}metadata.json"
    npz_path = session_dir / f"{prefix}samples.npz"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with np.load(npz_path) as data:
        npz = {k: data[k] for k in data.files}
    return RawSession(session_dir=session_dir, board_mac=board_mac, metadata=metadata, npz=npz)


def dominant_channel(npz: dict) -> int | None:
    if "channel_primary" not in npz or len(npz["channel_primary"]) == 0:
        return None
    values, counts = np.unique(npz["channel_primary"], return_counts=True)
    return int(values[np.argmax(counts)])


def decode_by_bucket(npz: dict) -> dict[int, dict]:
    csi_len = npz["csi_len"]
    csi_offset = npz["csi_offset"].astype(np.int64)
    csi_flat = npz["csi_flat"]
    buckets: dict[int, dict] = {}
    for length in np.unique(csi_len):
        length = int(length)
        idx = np.flatnonzero(csi_len == length)
        n_sub = length // 2
        offs = csi_offset[idx]
        gather = offs[:, None] + np.arange(length)[None, :]
        raw = csi_flat[gather].astype(np.float32)
        imag, real = raw[:, 0::2], raw[:, 1::2]
        buckets[length] = {
            "packet_indices": idx, "n_subcarriers": n_sub,
            "amplitude": np.hypot(real, imag).astype(np.float32),
            "phase": np.arctan2(imag, real).astype(np.float32),
        }
    return buckets


def dominant_bucket(npz: dict) -> tuple[int, dict] | None:
    if len(npz.get("csi_len", [])) == 0:
        return None
    buckets = decode_by_bucket(npz)
    lens, counts = np.unique(npz["csi_len"], return_counts=True)
    dom_len = int(lens[np.argmax(counts)])
    return dom_len, buckets[dom_len]


def decode_session_amplitude_phase(session_dir: Path, board_mac: str | None = None) -> dict | None:
    """One-shot convenience: load + decode a (session, receiver) unit's dominant bucket, returning
    amplitude/phase/rssi arrays plus rssi/channel/n_subcarriers metadata, or None if unusable."""
    session = load_raw_session(session_dir, board_mac=board_mac)
    result = dominant_bucket(session.npz)
    if result is None:
        return None
    _, bucket = result
    idx = bucket["packet_indices"]
    return {
        "amplitude": bucket["amplitude"], "phase": bucket["phase"], "n_subcarriers": bucket["n_subcarriers"],
        "rssi": session.npz["rssi"][idx].astype(np.float32) if "rssi" in session.npz else None,
        "channel": dominant_channel(session.npz), "metadata": session.metadata,
    }
