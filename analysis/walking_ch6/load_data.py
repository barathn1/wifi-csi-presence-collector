"""Build the restricted window index + feature matrix for the walking-only, channel-6-only
identifiability study -- the gait-dynamics counterpart to analysis/standing_ch6, same methodology, so
the two are directly comparable.

Scope (mirrors standing_ch6, motion filter swapped):
- motion == "walking" (occupied sessions) OR label == "none" (empty room, no motion field at all)
- only the one receiver board present on every date (`FIXED_BOARD`) -- confirmed via manifest that this
  board recorded every session across every collection date, so board/viewpoint is held fixed.
- channel 6 / 20MHz sessions ONLY -- detected PER SESSION from the actual packet data
  (`ml.data_pipeline.decode_csi.channel_width_summary`), not from a hardcoded date list. Earlier passes
  assumed (per data/README.md's "last 3 days" note) that only 2026-09-21/-22/-24 used channel 6 -- that
  turned out to be wrong: 2026-09-16 and -17 also used channel 6/20MHz throughout, and 2026-09-15 is a
  MIX of channel 6 and channel 11 sessions on the same date. So this filters session-by-session on the
  measured channel/bandwidth, which is the only way to get this right -- 6 total channel-6 dates
  (09-15, -16, -17, -21, -22, -24), not 3.

Reuses the existing decode/window/feature pipeline (ml.data_pipeline.*) rather than re-deriving CSI
parsing -- see that package for the wire-format details.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT, channel_width_summary, load_session
from ml.data_pipeline.features import build_feature_matrix, feature_names
from ml.data_pipeline.windowing import build_window_index, cache_session, load_manifest

FIXED_BOARD = "ac:27:6e:a5:5b:c8"
WINDOW_PACKETS = 200
STRIDE_PACKETS = 100

CACHE_DIR = Path(__file__).resolve().parent / "cache"
WINDOW_INDEX_PATH = CACHE_DIR / "window_index.csv"
FEATURES_PATH = CACHE_DIR / "features.npy"
CHANNEL_CACHE_PATH = CACHE_DIR / "session_channels.csv"


def detect_channel6_sessions(manifest: pd.DataFrame, force: bool = False) -> pd.Series:
    """Per-session (channel_primary, cwb) via the packet data itself -- returns a boolean Series aligned
    to `manifest.index`, True where that session is dominantly channel 6 / 20MHz (cwb=0)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CHANNEL_CACHE_PATH.exists() and not force:
        cached = pd.read_csv(CHANNEL_CACHE_PATH).set_index("session_dir")
    else:
        cached = pd.DataFrame(columns=["channel_primary", "cwb"]).set_index(pd.Index([], name="session_dir"))

    rows = {}
    for session_dir in manifest["session_dir"]:
        if session_dir in cached.index:
            rows[session_dir] = cached.loc[session_dir, ["channel_primary", "cwb"]].to_dict()
            continue
        session = load_session(REPO_ROOT / "data" / session_dir, board_mac=FIXED_BOARD)
        info = channel_width_summary(session.npz)
        rows[session_dir] = {"channel_primary": info["channel_primary"], "cwb": info["cwb"]}

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "session_dir"
    result.to_csv(CHANNEL_CACHE_PATH)

    aligned = result.reindex(manifest["session_dir"]).reset_index(drop=True)
    return (aligned["channel_primary"] == 6) & (aligned["cwb"] == 0)


def restricted_manifest() -> pd.DataFrame:
    manifest = load_manifest()
    manifest = manifest.copy()
    manifest["date"] = manifest["session_dir"].str.split("/").str[1]
    base_mask = (
        (manifest["board_mac"] == FIXED_BOARD)
        & ((manifest["motion"] == "walking") | (manifest["label"] == "none"))
    )
    candidates = manifest[base_mask].reset_index(drop=True)
    is_ch6 = detect_channel6_sessions(candidates)
    out = candidates[is_ch6.values].reset_index(drop=True)
    return out


def add_identity_column(window_index: pd.DataFrame) -> pd.DataFrame:
    """`identity`: anjali / barath / stranger:<name> / empty_room -- the label actually used for
    person-identifiability (as opposed to `label`, which is the access-control authorized/unauthorized/
    none framing)."""
    window_index = window_index.copy()

    def _identity(row):
        if row["label"] == "none":
            return "empty_room"
        if row["label"] == "authorized":
            return row["person_id"]
        return f"stranger:{row['person_id']}"

    window_index["identity"] = window_index.apply(_identity, axis=1)
    return window_index


def build(force: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if WINDOW_INDEX_PATH.exists() and FEATURES_PATH.exists() and not force:
        window_index = pd.read_csv(WINDOW_INDEX_PATH)
        X = np.load(FEATURES_PATH)
        return window_index, X

    manifest = restricted_manifest()
    print(f"restricted manifest: {len(manifest)} session rows")
    print(manifest.groupby(["date", "label", "person_id"], dropna=False).size())

    for _, row in manifest.iterrows():
        cache_session(row["session_dir"], mode="resampled", board_mac=FIXED_BOARD)

    window_index = build_window_index(
        manifest=manifest,
        mode="resampled",
        window_packets=WINDOW_PACKETS,
        stride_packets=STRIDE_PACKETS,
        board_mac=FIXED_BOARD,
    )
    window_index = add_identity_column(window_index)
    window_index.to_csv(WINDOW_INDEX_PATH, index=False)
    print(f"built {len(window_index)} windows -> {WINDOW_INDEX_PATH}")
    print(window_index.groupby(["date", "identity"]).size())

    X = build_feature_matrix(window_index)
    np.save(FEATURES_PATH, X)
    print(f"feature matrix: {X.shape} -> {FEATURES_PATH}")
    assert X.shape[1] == len(feature_names())
    return window_index, X


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    build(force=args.force)
