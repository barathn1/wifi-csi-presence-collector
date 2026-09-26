"""Train the presence gate (occupied vs. empty room) used by live_inference.py to decide WHEN to even
start the identity clock. Presence is a much higher-SNR, easier problem than identity (see
FINDINGS.md/DEPLOYMENT.md), so a single SVM on handcrafted stats is enough -- no need for the deep
branch here. Trained on ALL 6 channel-6 days pooled (walking sessions -- any person -- vs empty-room
sessions), same "final deployable, not a cross-day estimate" caveat as train_final.py.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cleaning import hampel_filter_amplitude
from data import build_session_table
from decode import decode_dominant_bucket, load_npz
from features import window_features
from subcarrier_mask import keep_mask
from windowing import STRIDE_PACKETS, WINDOW_PACKETS, make_windows

KEEP = keep_mask()
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
MAX_WINDOWS_PER_SESSION = 150
RNG = np.random.default_rng(0)


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    sessions = build_session_table(motion="walking", include_empty_room=True)
    print(f"{len(sessions)} sessions: {(sessions['label'] != 'none').sum()} occupied / "
          f"{(sessions['label'] == 'none').sum()} empty-room")

    stats_rows, y_rows = [], []
    for _, row in sessions.iterrows():
        npz = load_npz(Path(row["npz_path"]))
        bucket = decode_dominant_bucket(npz)
        amp, phase, rssi = bucket["amplitude"], bucket["phase"], bucket["rssi"]
        amp, _ = hampel_filter_amplitude(amp)
        amp, phase = amp[:, KEEP], phase[:, KEEP]

        windows = list(make_windows(amp, phase, rssi, WINDOW_PACKETS, STRIDE_PACKETS))
        if len(windows) > MAX_WINDOWS_PER_SESSION:
            keep = RNG.choice(len(windows), size=MAX_WINDOWS_PER_SESSION, replace=False)
            windows = [windows[i] for i in sorted(keep)]

        y = 0 if row["label"] == "none" else 1
        for start, end in windows:
            stats_rows.append(window_features(amp[start:end], phase[start:end], rssi[start:end]))
            y_rows.append(y)
        print(f"  {row['session_dir']}: {len(windows)} windows, y={y}")

    X = np.stack(stats_rows).astype(np.float32)
    y = np.array(y_rows)
    print(f"\n{len(y)} windows total, {y.sum()} occupied / {(y == 0).sum()} empty")

    pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                                     probability=True, class_weight="balanced",
                                                     max_iter=3000))
    pipeline.fit(X, y)
    train_acc = (pipeline.predict(X) == y).mean()
    print(f"presence SVM: pooled-set accuracy {train_acc:.3f} (NOT a cross-day estimate -- see "
          f"analysis/*_ch6/environment_control.py for the honest leave-one-day-out presence numbers, "
          f"77-96%, that justify shipping this)")

    path = CHECKPOINT_DIR / "presence_final.joblib"
    joblib.dump(pipeline, path)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
