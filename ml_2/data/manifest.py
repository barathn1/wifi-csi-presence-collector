"""Scans data/ directly and builds a manifest where each row is one (session, receiver) unit --
multi-receiver sessions contribute one row PER receiver (each ESP treated as its own independent
sample, not collapsed to a single 'representative' board), and the channel-6 filter is computed fresh
from each row's own raw per-packet `channel_primary` field, not read off any prior finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_2.data.decode import DATA_DIR, dominant_channel, list_receiver_macs, load_raw_session

TARGET_CHANNEL = 6
LABELS = ("authorized", "unauthorized", "none")


def build_full_manifest() -> pd.DataFrame:
    """Every (session, receiver) unit found under data/, with its own detected channel -- no filtering
    yet, so callers can inspect the full channel distribution before deciding what to keep."""
    rows = []
    for label in LABELS:
        label_dir = DATA_DIR / label
        if not label_dir.exists():
            continue
        for date_dir in sorted(label_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for session_dir in sorted(date_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                for mac in list_receiver_macs(session_dir):
                    try:
                        session = load_raw_session(session_dir, board_mac=mac)
                    except Exception as exc:
                        print(f"  skip {session_dir} ({mac}): {exc}")
                        continue
                    npz = session.npz
                    if "csi_len" not in npz or len(npz["csi_len"]) == 0:
                        continue
                    channel = dominant_channel(npz)
                    meta = session.metadata
                    dom_len = int(np.unique(npz["csi_len"], return_counts=True)[0]
                                  [np.argmax(np.unique(npz["csi_len"], return_counts=True)[1])])
                    rows.append({
                        "session_dir": str(session_dir.relative_to(DATA_DIR)),
                        # `receiver_mac`: always the real board MAC (from metadata, for single-receiver
                        # sessions where `mac` is None) -- used for grouping/baselines. `is_prefixed`
                        # (mac is not None) is the separate bit that decides whether the on-disk files
                        # actually have a MAC-prefixed name -- single-receiver sessions never do, even
                        # though we still know and record their board's MAC here.
                        "receiver_mac": mac if mac else str(meta.get("board_mac", "single")),
                        "is_prefixed": mac is not None,
                        "label": label,
                        "person_id": meta.get("person_id", "") or "",
                        "date": date_dir.name,
                        "channel": channel,
                        "n_packets": int(len(npz["csi_len"])),
                        "dominant_csi_len": dom_len,
                        "n_receivers_in_session": None,  # filled in below
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    n_recv = df.groupby("session_dir")["receiver_mac"].transform("count")
    df["n_receivers_in_session"] = n_recv
    return df


def build_channel6_manifest(min_packets: int = 50) -> pd.DataFrame:
    full = build_full_manifest()
    print(f"full manifest: {len(full)} (session,receiver) rows across "
          f"{full['session_dir'].nunique()} sessions")
    print("channel distribution (rows):", full["channel"].value_counts().to_dict())
    print("channel distribution (by date):")
    print(full.groupby("date")["channel"].agg(lambda s: s.value_counts().to_dict()))

    ch6 = full[(full["channel"] == TARGET_CHANNEL) & (full["n_packets"] >= min_packets)].reset_index(drop=True)
    print(f"\nchannel-{TARGET_CHANNEL}-only, >= {min_packets} packets: {len(ch6)} (session,receiver) rows "
          f"across {ch6['session_dir'].nunique()} sessions, dates={sorted(ch6['date'].unique())}")
    print(f"label counts: {ch6['label'].value_counts().to_dict()}")
    print(f"dominant_csi_len distribution: {ch6['dominant_csi_len'].value_counts().to_dict()}")
    return ch6


if __name__ == "__main__":
    build_channel6_manifest()
