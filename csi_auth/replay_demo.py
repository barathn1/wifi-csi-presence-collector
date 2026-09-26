"""Feed a REAL recorded session through LiveIdentitySession exactly as a live stream would see it --
no hardware needed, since this is just replaying already-collected packets in their original order
with their original timing. Prints the on-screen status at every window, so you can see exactly what
the demo would have shown, and when, for a real Anjali/Barath/stranger/empty-room recording.

    python replay_demo.py <samples.npz> [reveal_after_s] [model_name]

    model_name: cnn_attention (default, recommended) | cnn_bilstm | svm -- see DEPLOYMENT.md.
    IMPORTANT: use the FIXED_BOARD file for the session (see data.py) -- e.g.
    ac276ea55bc8_samples.npz, not a4cb8fd452b0_samples.npz or ac276ea2f278_samples.npz. Loading the
    wrong receiver's file produced a sustained, high-confidence false positive in testing (see
    DEPLOYMENT.md) -- the models only know this one board's noise floor.

    python replay_demo.py data/authorized/2026-09-24/20260924_164118_anjali/ac276ea55bc8_samples.npz
    python replay_demo.py data/authorized/2026-09-24/20260924_165513_barath/ac276ea55bc8_samples.npz 35 svm
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from cleaning import hampel_filter_amplitude
from decode import decode_dominant_bucket, load_npz, monotonic_elapsed_seconds
from live_inference import LiveIdentitySession
from subcarrier_mask import keep_mask
from windowing import STRIDE_PACKETS, WINDOW_PACKETS, make_windows

KEEP = keep_mask()


def replay(npz_path: str, reveal_after_s: float = 35.0, verbose_every: int = 5,
           identity_model_name: str = "cnn_attention"):
    npz = load_npz(Path(npz_path))
    bucket = decode_dominant_bucket(npz)
    amp, phase, rssi = bucket["amplitude"], bucket["phase"], bucket["rssi"]
    amp, _ = hampel_filter_amplitude(amp)
    amp, phase = amp[:, KEEP], phase[:, KEEP]
    elapsed_all = monotonic_elapsed_seconds(bucket["device_time_us"])

    session = LiveIdentitySession(reveal_after_s=reveal_after_s, identity_model_name=identity_model_name)
    print(f"replaying {npz_path} ({amp.shape[0]} packets) with model={identity_model_name!r}\n")

    last_display = None
    for i, (start, end) in enumerate(make_windows(amp, phase, rssi, WINDOW_PACKETS, STRIDE_PACKETS)):
        elapsed_s = elapsed_all[end - 1]
        status = session.on_window(amp[start:end], phase[start:end], rssi[start:end], elapsed_s)

        # print on every state change, plus every `verbose_every` windows while accumulating
        if status["display"] != last_display or (status["display"] == "accumulating (silent)" and i % verbose_every == 0):
            print(f"  t={elapsed_s:6.1f}s  window {i:3d}  -> {status}")
        last_display = status["display"]

    print(f"\nfinal status: {session.status()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    reveal_s = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
    model_name = sys.argv[3] if len(sys.argv) > 3 else "cnn_attention"
    replay(sys.argv[1], reveal_after_s=reveal_s, identity_model_name=model_name)
