"""Scan data/manifest.csv (a plain CSV data index, not pipeline code) and the raw samples.npz files to
build the session list for this study: walking-motion sessions, channel-6/20MHz only (detected
per-session from the packet data itself), one fixed receiver board.

DATA_ROOT points at wifi-csi-presence-collector/data -- we read its data files directly and do not
import any of its Python code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from decode import dominant_channel, load_npz

DATA_ROOT = Path(r"C:\Users\av9146\Desktop\wifi-sense\wifi-csi-presence-collector\data")


def session_npz_path(session_dir: str, board_mac: str) -> Path:
    """Multi-receiver sessions (2026-09-21 onward) prefix files with the board's MAC, colons stripped;
    single-receiver sessions use the plain filename. Try prefixed first, fall back to plain."""
    d = DATA_ROOT / session_dir
    prefixed = d / f"{board_mac.replace(':', '').lower()}_samples.npz"
    return prefixed if prefixed.exists() else d / "samples.npz"


def find_fixed_board(manifest: pd.DataFrame) -> str:
    """The one board_mac value present on every collection date -- verified from the manifest itself,
    not assumed."""
    manifest = manifest.copy()
    manifest["date"] = manifest["session_dir"].str.split("/").str[1]
    per_date_boards = manifest.groupby("date")["board_mac"].apply(set)
    common = set.intersection(*per_date_boards.tolist())
    macs = [m for m in common if ":" in str(m)]  # excludes any "192.168.x.x" transport-address rows
    assert len(macs) == 1, f"expected exactly one board on every date, got {macs}"
    return macs[0]


def build_session_table(motion: str = "walking") -> pd.DataFrame:
    manifest = pd.read_csv(DATA_ROOT / "manifest.csv")
    manifest["date"] = manifest["session_dir"].str.split("/").str[1]
    board = find_fixed_board(manifest)

    candidates = manifest[
        (manifest["board_mac"] == board) & (manifest["motion"] == motion)
    ].reset_index(drop=True)

    rows = []
    for _, row in candidates.iterrows():
        npz_path = session_npz_path(row["session_dir"], board)
        if not npz_path.exists():
            continue
        npz = load_npz(npz_path)
        channel, cwb = dominant_channel(npz)
        if channel != 6 or cwb != 0:
            continue
        rows.append({
            "session_dir": row["session_dir"],
            "npz_path": str(npz_path),
            "date": row["date"],
            "label": row["label"],
            "person_id": row["person_id"],
        })
    out = pd.DataFrame(rows)
    out["auth"] = out["person_id"].isin(["anjali", "barath"]).astype(int)
    return out


if __name__ == "__main__":
    table = build_session_table()
    print(f"{len(table)} channel-6 walking sessions on the fixed board")
    print(table.groupby(["date", "auth", "person_id"]).size())
