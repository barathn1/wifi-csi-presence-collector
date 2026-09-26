"""Build the restricted window index + feature matrix for the standing-only, channel-6-only
identifiability study.

Scope (deliberately narrow, per the user's "start with standing data first, channel 6 only" request):
- motion == "standing" (occupied sessions) OR label == "none" (empty room, no motion field at all)
- only the one receiver board present on every date (`FIXED_BOARD`) -- avoids conflating a second/third
  receiver's viewpoint with person identity
- only the three channel-6/20MHz dates (2026-09-21, -22, -24) -- per data/README.md, every earlier date
  used a different channel/bandwidth, so pooling them in would confound "person" with "PHY config".

Reuses the existing decode/window/feature pipeline (ml.data_pipeline.*) rather than re-deriving CSI
parsing -- see that package for the wire-format details.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.features import build_feature_matrix, feature_names
from ml.data_pipeline.windowing import build_window_index, cache_session, load_manifest

FIXED_BOARD = "ac:27:6e:a5:5b:c8"
CH6_DATES = ["2026-09-21", "2026-09-22", "2026-09-24"]
WINDOW_PACKETS = 200
STRIDE_PACKETS = 100

CACHE_DIR = Path(__file__).resolve().parent / "cache"
WINDOW_INDEX_PATH = CACHE_DIR / "window_index.csv"
FEATURES_PATH = CACHE_DIR / "features.npy"


def restricted_manifest() -> pd.DataFrame:
    manifest = load_manifest()
    manifest = manifest.copy()
    manifest["date"] = manifest["session_dir"].str.split("/").str[1]
    mask = (
        manifest["date"].isin(CH6_DATES)
        & (manifest["board_mac"] == FIXED_BOARD)
        & ((manifest["motion"] == "standing") | (manifest["label"] == "none"))
    )
    out = manifest[mask].reset_index(drop=True)
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
