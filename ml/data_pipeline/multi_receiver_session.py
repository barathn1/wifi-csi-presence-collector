"""Load sessions from the 2026-09-21+ multi-receiver collection format: instead of one
`metadata.json`/`samples.npz` pair per session directory, each of the (up to 3) simultaneous ESP32
receivers writes its OWN `<mac_no_colons>_metadata.json` / `<mac_no_colons>_samples.npz` pair into the
same session directory (one physical walk/stand, several receivers watching it from different
positions). `decode_csi.load_session` assumes the older single-receiver naming and can't read these at
all -- this module is the multi-receiver-aware equivalent, otherwise reusing every validated piece of
the single-receiver pipeline unchanged (`select_dominant_packets`, `decode_session_by_bucket`, the
Session NamedTuple's `.npz`/`.metadata` shape) so callers built around one receiver need zero changes,
just a different way to enumerate "which receiver(s) does this session have data from".

Not every receiver captured a valid `samples.npz` for every session in practice -- e.g. one receiver's
config negotiated a bad PHY combo, one dropped its stimulus connection mid-recording, or (as found
while inspecting the freshly-collected 2026-09-21 set) a receiver's npz simply hadn't finished writing/
syncing yet. `list_receivers` reports exactly which receiver MACs have a *usable* npz for a session,
never assumes all 3 are present.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.data_pipeline.decode_csi import Session


def list_receivers(session_dir: Path) -> list[str]:
    """Receiver MACs (colon-separated) that have BOTH a metadata.json and a samples.npz in this
    session directory -- i.e. are actually usable, not just present as a metadata-only stub."""
    receivers = []
    for meta_path in sorted(session_dir.glob("*_metadata.json")):
        prefix = meta_path.name[: -len("_metadata.json")]
        npz_path = session_dir / f"{prefix}_samples.npz"
        if not npz_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        receivers.append(meta.get("board_mac", prefix))
    return receivers


def load_multi_receiver_session(session_dir: Path, receiver_mac: str) -> Session:
    """Same (metadata, npz) shape as `decode_csi.load_session`, for one specific receiver's files
    within a multi-receiver session directory. `receiver_mac` must be one of `list_receivers`'
    output (a colon-separated MAC, matched against each file's own recorded `board_mac` -- not
    assumed from the filename prefix, since one observed 2026-09-21 session used raw IP-address
    strings as the filename prefix instead of a MAC)."""
    for meta_path in sorted(session_dir.glob("*_metadata.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("board_mac") != receiver_mac:
            continue
        prefix = meta_path.name[: -len("_metadata.json")]
        npz_path = session_dir / f"{prefix}_samples.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} missing (metadata claims board_mac={receiver_mac})")
        with np.load(npz_path) as data:
            npz = {k: data[k] for k in data.files}
        return Session(session_dir=session_dir, metadata=meta, npz=npz)
    raise ValueError(f"no metadata file in {session_dir} has board_mac == {receiver_mac}")
