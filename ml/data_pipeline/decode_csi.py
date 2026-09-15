"""Decode the ragged CSI wire format from a collected session into amplitude/phase arrays.

`samples.npz` stores CSI as a flat, ragged buffer: `csi_flat[csi_offset[i] : csi_offset[i]+csi_len[i]]`
holds `csi_len[i] // 2` (imag, real) int8 pairs for packet i (one pair per subcarrier). `csi_len` is not
constant across a session -- different 802.11 frame types (legacy/HT, 20/40MHz) carry different
subcarrier counts. This module buckets packets by `csi_len` and vectorizes the decode within each bucket
(same length => can gather into a dense 2D array in one shot, no per-packet Python loop).

    python3 -m ml.data_pipeline.decode_csi data/authorized/2026-09-09/20260909_152250_anjali --sample 5

cross-checks one decoded sample against `collector/inspect_npz.py`'s raw (imag,real) pair printout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


class Session(NamedTuple):
    session_dir: Path
    metadata: dict
    npz: dict  # loaded arrays, kept in memory (sessions are tens of MB, fine)


def resolve_session_dir(path: Path) -> Path:
    if path.name == "samples.npz":
        return path.parent
    return path


def load_session(session_dir: Path) -> Session:
    session_dir = resolve_session_dir(Path(session_dir))
    meta_path = session_dir / "metadata.json"
    npz_path = session_dir / "samples.npz"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    with np.load(npz_path) as data:
        npz = {k: data[k] for k in data.files}
    return Session(session_dir=session_dir, metadata=metadata, npz=npz)


def decode_session_by_bucket(npz: dict) -> dict[int, dict]:
    """Group packets by csi_len and vectorize the (imag,real) int8 -> amplitude/phase decode.

    Returns {csi_len_bytes: {"packet_indices", "n_subcarriers", "amplitude", "phase"}}.
    amplitude/phase arrays have shape (n_packets_in_bucket, n_subcarriers).
    """
    csi_len = npz["csi_len"]
    csi_offset = npz["csi_offset"].astype(np.int64)
    csi_flat = npz["csi_flat"]

    buckets: dict[int, dict] = {}
    for length in np.unique(csi_len):
        length = int(length)
        idx = np.flatnonzero(csi_len == length)
        n_sub = length // 2
        offs = csi_offset[idx]
        gather_idx = offs[:, None] + np.arange(length)[None, :]
        raw = csi_flat[gather_idx].astype(np.float32)  # (n_packets, length)
        imag = raw[:, 0::2]
        real = raw[:, 1::2]
        buckets[length] = {
            "packet_indices": idx,
            "n_subcarriers": n_sub,
            "amplitude": np.hypot(real, imag).astype(np.float32),
            "phase": np.arctan2(imag, real).astype(np.float32),
        }
    return buckets


def decode_one_packet(npz: dict, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference (non-vectorized) single-packet decode -- used for verification/spot checks."""
    off = int(npz["csi_offset"][index])
    length = int(npz["csi_len"][index])
    raw = npz["csi_flat"][off:off + length]
    imag = raw[0::2].astype(np.float32)
    real = raw[1::2].astype(np.float32)
    complex_vals = real + 1j * imag
    return complex_vals, np.hypot(real, imag), np.arctan2(imag, real)


def csi_len_distribution(npz: dict) -> dict[int, int]:
    lens, counts = np.unique(npz["csi_len"], return_counts=True)
    return {int(l): int(c) for l, c in zip(lens, counts)}


def decode_one_sample(csi_data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode one LIVE packet's raw CSI bytes into (amplitude, phase) -- same (imag,real) int8-pair
    math as decode_session_by_bucket/decode_one_packet, just for a single live-arriving packet rather
    than an already-recorded session array. Shared by ml/inference/live_infer.py and
    ml/visualization/player_server.py (each used to keep its own copy of this)."""
    raw = np.frombuffer(csi_data, dtype=np.int8).astype(np.float32)
    imag, real = raw[0::2], raw[1::2]
    return np.hypot(real, imag).astype(np.float32), np.arctan2(imag, real).astype(np.float32)


def channel_2ghz_freq_mhz(channel: int) -> int:
    return 2412 + 5 * (channel - 1)


def _summarize_channel_combo(combo_counts: dict[tuple[int, int, int], int]) -> dict:
    """Shared core of channel_width_summary/channel_width_summary_live: given packet counts per
    (cwb, channel_primary, channel_secondary) combo, pick the dominant one and describe it in human-
    readable channel/bandwidth/frequency-span terms. `cwb`: 0=20MHz (HT20), 1=40MHz (HT40).
    `channel_secondary` (ESP-IDF wifi_second_chan_t): 0=NONE, 1=ABOVE, 2=BELOW."""
    (dom_cwb, dom_primary, dom_secondary), dom_count = max(combo_counts.items(), key=lambda kv: kv[1])
    total = sum(combo_counts.values())

    primary_center = channel_2ghz_freq_mhz(dom_primary)
    if dom_cwb == 0:
        desc = f"20MHz (HT20), channel {dom_primary}, {primary_center - 10}-{primary_center + 10} MHz"
    else:
        direction = {0: "NONE", 1: "ABOVE", 2: "BELOW"}.get(dom_secondary, f"?{dom_secondary}")
        if dom_secondary == 1:
            span = (primary_center - 10, primary_center + 30)
        elif dom_secondary == 2:
            span = (primary_center - 30, primary_center + 10)
        else:
            span = (primary_center - 10, primary_center + 10)
        desc = f"40MHz (HT40), channel {dom_primary}+{direction}, {span[0]}-{span[1]} MHz"

    return {"cwb": dom_cwb, "channel_primary": dom_primary, "channel_secondary": dom_secondary,
            "description": desc, "fraction_of_packets": float(dom_count / total)}


def channel_width_summary(npz: dict) -> dict:
    """Dominant (cwb, channel_primary, channel_secondary) combo for an already-recorded session,
    translated into a human-readable description -- no router access needed, everything here comes
    straight from the per-packet rx_ctrl fields the firmware already records. See
    channel_width_summary_live() for the equivalent check against a live, in-progress stream."""
    cwb = npz["cwb"]
    ch_primary = npz["channel_primary"]
    ch_secondary = npz["channel_secondary"]
    combos, counts = np.unique(
        np.stack([cwb, ch_primary, ch_secondary], axis=1), axis=0, return_counts=True,
    )
    combo_counts = {(int(c[0]), int(c[1]), int(c[2])): int(n) for c, n in zip(combos, counts)}
    return _summarize_channel_combo(combo_counts)


def channel_width_summary_live(combo_counts: dict[tuple[int, int, int], int]) -> dict:
    """Same as channel_width_summary but for counts accumulated incrementally from a live stream (e.g.
    a collections.Counter keyed by (cwb, channel_primary, channel_secondary) that
    ml/inference/live_infer.py updates per accepted packet) instead of a full recorded session array."""
    return _summarize_channel_combo(dict(combo_counts))


def list_sessions(label: str | None = None) -> list[Path]:
    """All session dirs under data/, optionally filtered to one label (authorized/unauthorized/none)."""
    labels = [label] if label else ["authorized", "unauthorized", "none"]
    sessions = []
    for lbl in labels:
        label_dir = DATA_DIR / lbl
        if not label_dir.exists():
            continue
        for date_dir in sorted(label_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for session_dir in sorted(date_dir.iterdir()):
                if (session_dir / "samples.npz").exists():
                    sessions.append(session_dir)
    return sessions


def _verify_against_inspect_npz(session_dir: Path, sample_index: int) -> None:
    session = load_session(session_dir)
    complex_vals, amplitude, phase = decode_one_packet(session.npz, sample_index)

    off = int(session.npz["csi_offset"][sample_index])
    length = int(session.npz["csi_len"][sample_index])
    raw = session.npz["csi_flat"][off:off + length]
    pairs = [(int(a), int(b)) for a, b in zip(raw[0::2], raw[1::2])]

    print(f"session: {session_dir}")
    print(f"sample #{sample_index}: csi_len={length} bytes -> {length // 2} subcarriers")
    print(f"raw (imag,real) pairs (matches collector/inspect_npz.py --sample {sample_index}):")
    print(f"  {pairs[:8]}{' ...' if len(pairs) > 8 else ''}")
    print(f"decoded amplitude[:8]: {np.round(amplitude[:8], 3)}")
    print(f"decoded phase[:8]:     {np.round(phase[:8], 3)}")

    recon_imag = np.round(complex_vals.imag).astype(int)
    recon_real = np.round(complex_vals.real).astype(int)
    ok = all(recon_imag[i] == pairs[i][0] and recon_real[i] == pairs[i][1] for i in range(len(pairs)))
    print(f"round-trip check vs raw pairs: {'OK' if ok else 'MISMATCH'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="session directory")
    p.add_argument("--sample", type=int, default=0, help="packet index to decode and verify")
    p.add_argument("--buckets", action="store_true", help="print csi_len bucket distribution instead")
    p.add_argument("--channel", action="store_true",
                   help="print dominant channel/bandwidth (20MHz vs 40MHz, which channel, frequency "
                        "span) -- no router access needed, reads straight from samples.npz")
    args = p.parse_args()

    session_dir = resolve_session_dir(Path(args.path))
    if args.channel:
        session = load_session(session_dir)
        info = channel_width_summary(session.npz)
        print(f"{session_dir}: {info['description']} ({100 * info['fraction_of_packets']:.1f}% of packets)")
    elif args.buckets:
        session = load_session(session_dir)
        dist = csi_len_distribution(session.npz)
        n = sum(dist.values())
        print(f"{session_dir}: {n} packets")
        for length, count in sorted(dist.items()):
            print(f"  csi_len={length:4d} bytes ({length // 2:3d} subcarriers): "
                  f"{count:7d} ({100 * count / n:.1f}%)")
    else:
        _verify_against_inspect_npz(session_dir, args.sample)


if __name__ == "__main__":
    main()
