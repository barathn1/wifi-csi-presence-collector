"""Build the Doppler-spectrum feature matrix for auth-vs-non-auth (all walking sessions -- Anjali,
Barath, and every stranger), parallel to dataset.py's handcrafted-stats matrix but frequency-domain
instead of time-domain. Kept as a separate cache/script rather than folding into dataset.py so the
existing, already-validated pipeline is untouched -- this is a new, still-being-tested feature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cleaning import hampel_filter_amplitude
from data import build_session_table
from decode import decode_dominant_bucket, load_npz, monotonic_elapsed_seconds
from stft_features import window_doppler_spectrum
from subcarrier_mask import keep_mask
from windowing import make_windows

KEEP = keep_mask()
CACHE_DIR = Path(__file__).resolve().parent / "cache"
MAX_WINDOWS_PER_SESSION = 80

# A DIFFERENT, LONGER window than the 200-packet one used everywhere else in this package (time-domain
# stats/CNN branches) -- deliberately not windowing.WINDOW_PACKETS. At this dataset's observed capture
# rates (200-500 Hz), 200 packets is as little as ~0.3-0.4s: far too short to resolve stride-rate
# Doppler content (~1-2 Hz needs >=1-2s just to fit a full gait cycle, and frequency resolution =
# 1/duration -- a 0.34s window has ~3 Hz resolution, unable to distinguish a 1 Hz from a 2 Hz signal at
# all). 1000 packets/500 stride guarantees >=2s even at the fastest observed rate.
STFT_WINDOW_PACKETS = 1000
STFT_STRIDE_PACKETS = 500
RNG = np.random.default_rng(0)


def build_stft_dataset(force: bool = False):
    CACHE_DIR.mkdir(exist_ok=True)
    table_path = CACHE_DIR / "window_table_stft.csv"
    feat_path = CACHE_DIR / "X_doppler.npy"
    if not force and table_path.exists() and feat_path.exists():
        return pd.read_csv(table_path), np.load(feat_path)

    sessions = build_session_table(motion="walking")
    print(f"{len(sessions)} sessions ({sessions['auth'].sum()} auth / "
          f"{(~sessions['auth'].astype(bool)).sum()} non-auth)")

    rows, feat_rows = [], []
    for _, srow in sessions.iterrows():
        npz = load_npz(Path(srow["npz_path"]))
        bucket = decode_dominant_bucket(npz)
        amp, phase, rssi = bucket["amplitude"], bucket["phase"], bucket["rssi"]
        amp, _ = hampel_filter_amplitude(amp)
        amp = amp[:, KEEP]
        elapsed_s = monotonic_elapsed_seconds(bucket["device_time_us"])

        windows = list(make_windows(amp, phase, rssi, STFT_WINDOW_PACKETS, STFT_STRIDE_PACKETS))
        if len(windows) > MAX_WINDOWS_PER_SESSION:
            keep = RNG.choice(len(windows), size=MAX_WINDOWS_PER_SESSION, replace=False)
            windows = [windows[i] for i in sorted(keep)]

        for start, end in windows:
            feat_rows.append(window_doppler_spectrum(amp[start:end], elapsed_s[start:end]))
            rows.append({"session_dir": srow["session_dir"], "date": srow["date"],
                         "auth": srow["auth"], "person_id": srow["person_id"]})
        print(f"  {srow['session_dir']}: {len(windows)} windows")

    window_table = pd.DataFrame(rows)
    X = np.stack(feat_rows).astype(np.float32)

    window_table.to_csv(table_path, index=False)
    np.save(feat_path, X)
    print(f"built {len(window_table)} windows, X shape {X.shape} -> {CACHE_DIR}")
    return window_table, X


if __name__ == "__main__":
    wt, X = build_stft_dataset(force=True)
    print(X.shape)
    print(wt.groupby(["date", "auth"]).size())
