"""Build the full window-level dataset: handcrafted feature matrix (classical branch) + raw amplitude
sequences (deep branch) + labels, cached to disk so repeated runs don't re-decode.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cleaning import hampel_filter_amplitude
from data import build_session_table
from decode import decode_dominant_bucket, load_npz
from features import window_features
from subcarrier_mask import keep_mask
from windowing import STRIDE_PACKETS, WINDOW_PACKETS, make_windows

KEEP = keep_mask()  # drop the 19 verified-null guard/DC/pilot bins -- 128 -> 109 subcarriers

CACHE_DIR = Path(__file__).resolve().parent / "cache"
MAX_WINDOWS_PER_SESSION = 150  # overlapping windows are highly redundant; caps compute for the CNN branch
RNG = np.random.default_rng(0)


def build_dataset(motion: str = "walking", force: bool = False):
    CACHE_DIR.mkdir(exist_ok=True)
    table_path = CACHE_DIR / "window_table.csv"
    stats_path = CACHE_DIR / "X_stats.npy"
    seq_path = CACHE_DIR / "X_seq.npy"
    if not force and table_path.exists() and stats_path.exists() and seq_path.exists():
        return pd.read_csv(table_path), np.load(stats_path), np.load(seq_path)

    sessions = build_session_table(motion=motion)
    print(f"{len(sessions)} sessions ({sessions['auth'].sum()} auth / "
          f"{(~sessions['auth'].astype(bool)).sum()} non-auth)")

    rows, stats_rows, seq_rows = [], [], []
    for _, srow in sessions.iterrows():
        npz = load_npz(Path(srow["npz_path"]))
        bucket = decode_dominant_bucket(npz)
        assert bucket["n_subcarriers"] == 128, bucket["n_subcarriers"]
        amp, phase, rssi = bucket["amplitude"], bucket["phase"], bucket["rssi"]
        amp, frac_spikes = hampel_filter_amplitude(amp)
        if frac_spikes > 0.02:
            print(f"    {srow['session_dir']}: {100 * frac_spikes:.1f}% samples flagged as spikes")
        amp, phase = amp[:, KEEP], phase[:, KEEP]  # drop the 19 null guard/DC/pilot bins

        window_slices = list(make_windows(amp, phase, rssi, WINDOW_PACKETS, STRIDE_PACKETS))
        if len(window_slices) > MAX_WINDOWS_PER_SESSION:
            keep = RNG.choice(len(window_slices), size=MAX_WINDOWS_PER_SESSION, replace=False)
            window_slices = [window_slices[i] for i in sorted(keep)]

        for start, end in window_slices:
            a, p, r = amp[start:end], phase[start:end], rssi[start:end]
            stats_rows.append(window_features(a, p, r))
            seq_rows.append(a)  # raw amplitude sequence for the deep branch
            rows.append({"session_dir": srow["session_dir"], "date": srow["date"],
                         "auth": srow["auth"], "person_id": srow["person_id"]})
        print(f"  {srow['session_dir']}: {len(window_slices)} windows")

    window_table = pd.DataFrame(rows)
    X_stats = np.stack(stats_rows).astype(np.float32)
    X_seq = np.stack(seq_rows).astype(np.float32)

    window_table.to_csv(table_path, index=False)
    np.save(stats_path, X_stats)
    np.save(seq_path, X_seq)
    print(f"built {len(window_table)} windows -> {CACHE_DIR}")
    return window_table, X_stats, X_seq


if __name__ == "__main__":
    wt, Xs, Xq = build_dataset(force=True)
    print(Xs.shape, Xq.shape)
    print(wt.groupby(["date", "auth"]).size())
