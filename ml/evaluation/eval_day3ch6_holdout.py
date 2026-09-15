"""Offline evaluation of the `_day3ch6` checkpoints against sessions recorded AFTER that training run
finished -- genuinely held-out data (not just a different split of the same pool), and a chance to test
a specific hypothesis about the live-vs-offline discrepancy the user reported: training normalizes with
ONE `calibA` baseline pooled from all 6 training `none` sessions, but `ml/inference/live_infer.py`
recalibrates FRESH every run from a short live "stand outside" period. If that live baseline drifts from
the training one, every downstream z-score shifts -- this script measures how much, and evaluates
accuracy under both baselines side by side so the effect (if any) is visible, not assumed.

    python3 -m ml.evaluation.eval_day3ch6_holdout
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ml.data_pipeline.calibration import apply_variant_a, compute_day_baseline_from_sessions
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.windowing import build_window_index, load_manifest, load_window
from ml.inference.checkpoint import load_checkpoint
from ml.inference.live_calibration import compute_live_baseline
from ml.data_pipeline.windowing import cache_session

CHECKPOINT_DIR = REPO_ROOT / "ml/checkpoints"
TASKS = {
    "taskD_auth_vs_nonauth": "whofi_taskD_calibA_day3ch6.pt",
    "task0_presence": "whofi_task0_presence_calibA_day3ch6.pt",
    "taskE_motion_standing_vs_walking": "whofi_taskE_motion_calibA_day3ch6.pt",
}

# the 6 `none` sessions the training run's calibA baseline was pooled from
TRAIN_NONE_SESSIONS = [
    "none/2026-09-15/20260915_150333", "none/2026-09-15/20260915_150550",
    "none/2026-09-15/20260915_152301", "none/2026-09-15/20260915_152825",
    "none/2026-09-15/20260915_155425", "none/2026-09-15/20260915_155941",
]

# sessions recorded AFTER the training run finished (mtime-checked against the checkpoints)
HOLDOUT_SESSIONS = [
    "authorized/2026-09-15/20260915_180952_barath",
    "authorized/2026-09-15/20260915_181508_barath",
    "unauthorized/2026-09-15/20260915_175047_siva",
    "unauthorized/2026-09-15/20260915_175309_siva",
    "none/2026-09-15/20260915_175702",
    "none/2026-09-15/20260915_180705",
    "none/2026-09-15/20260915_182121",
    "none/2026-09-15/20260915_182340",
]
LIVE_CALIB_SESSION = "none/2026-09-15/20260915_175702"  # earliest holdout `none` -- stand-in for a fresh
# live "stand outside" calibration period, same role compute_live_baseline plays in live_infer.py
LIVE_CALIB_SECONDS = 60.0
ASSUMED_PACKET_RATE_HZ = 220.0  # rough, just to slice ~60s worth of packets for the live-style baseline


def ground_truth(task_name: str, row: pd.Series) -> int | None:
    if task_name == "taskD_auth_vs_nonauth":
        return int(row["label"] == "authorized")
    if task_name == "task0_presence":
        return int(row["label"] != "none")
    if task_name == "taskE_motion_standing_vs_walking":
        if row["label"] not in ("authorized", "unauthorized") or not row["motion"]:
            return None
        return 0 if row["motion"] == "standing" else 1  # matches alphabetical class sort
    raise ValueError(task_name)


def build_baselines():
    baseline_train = compute_day_baseline_from_sessions(TRAIN_NONE_SESSIONS, label="train_pooled")

    cache_path = cache_session(LIVE_CALIB_SESSION, mode="resampled")
    with np.load(cache_path) as d:
        amp, phase = d["amplitude"], d["phase"]
    n_calib_packets = min(len(amp), int(LIVE_CALIB_SECONDS * ASSUMED_PACKET_RATE_HZ))
    baseline_live = compute_live_baseline(amp[:n_calib_packets], phase[:n_calib_packets], label="live_fresh")

    drift_amp = float(np.mean(np.abs(baseline_train.amp_mean - baseline_live.amp_mean) / baseline_train.amp_std))
    drift_std_ratio = float(np.mean(baseline_live.amp_std / baseline_train.amp_std))
    print(f"[baseline drift] live calib used {n_calib_packets} packets from {LIVE_CALIB_SESSION}")
    print(f"[baseline drift] mean |train_mean - live_mean| / train_std = {drift_amp:.3f}  "
          f"(0 = identical baselines; >>0.3-0.5 means most windows' z-scores shift meaningfully)")
    print(f"[baseline drift] mean live_std / train_std = {drift_std_ratio:.3f}  "
          f"(1.0 = identical spread; far from 1.0 means the model sees rescaled inputs it never trained on)")
    return baseline_train, baseline_live


@torch.no_grad()
def predict_session(model, window_index: pd.DataFrame, session_dir: str, baseline) -> pd.DataFrame:
    rows = window_index[window_index["session_dir"] == session_dir].sort_values("start")
    probs = []
    for _, row in rows.iterrows():
        amp, phase, _ = load_window(row)
        amp_z, phase_z = apply_variant_a(amp, phase, baseline)
        amp_t = torch.from_numpy(amp_z.astype(np.float32)).unsqueeze(0)
        phase_t = torch.from_numpy(phase_z.astype(np.float32)).unsqueeze(0)
        logits = model(amp_t, phase_t)
        probs.append(torch.softmax(logits, dim=-1)[0, 1].item())
    out = rows.copy()
    out["proba"] = probs
    return out


def rolling_aggregate_accuracy(probs: np.ndarray, y_true: int, sizes=(1, 10, 30, 60, 90, 120)) -> dict:
    """y_true is a SINGLE label shared by every window in this session (session-level ground truth) --
    accuracy at aggregate size N = fraction of positions where the trailing-N-window rolling mean lands
    on the correct side of 0.5, once N windows are available."""
    out = {}
    for n in sizes:
        if len(probs) < n:
            out[n] = float("nan")
            continue
        agg = np.convolve(probs, np.ones(n) / n, mode="valid")
        pred = (agg >= 0.5).astype(int)
        out[n] = float((pred == y_true).mean())
    return out


def main() -> None:
    manifest = load_manifest()
    holdout_manifest = manifest[manifest["session_dir"].isin(HOLDOUT_SESSIONS)].reset_index(drop=True)
    assert len(holdout_manifest) == len(HOLDOUT_SESSIONS), \
        f"expected {len(HOLDOUT_SESSIONS)} holdout sessions, found {len(holdout_manifest)} in manifest"

    print(f"holdout sessions: {len(holdout_manifest)} "
          f"({holdout_manifest['label'].value_counts().to_dict()})")
    window_index = build_window_index(manifest=holdout_manifest, mode="resampled",
                                       window_packets=200, stride_packets=100)
    print(f"holdout window index: {len(window_index)} windows\n")

    baseline_train, baseline_live = build_baselines()

    checkpoints = {name: load_checkpoint(CHECKPOINT_DIR / fn) for name, fn in TASKS.items()}

    for task_name, ck in checkpoints.items():
        print(f"\n=== {task_name} ===")
        for _, srow in holdout_manifest.iterrows():
            session_dir = srow["session_dir"]
            y_true = ground_truth(task_name, srow)
            if y_true is None:
                continue
            for baseline_name, baseline in [("train_baseline", baseline_train), ("live_baseline", baseline_live)]:
                pred_df = predict_session(ck.model, window_index, session_dir, baseline)
                probs = pred_df["proba"].values
                inst_acc = float(((probs >= 0.5).astype(int) == y_true).mean())
                agg = rolling_aggregate_accuracy(probs, y_true)
                agg_str = "  ".join(f"@{n}w={v:.2f}" for n, v in agg.items())
                print(f"  {session_dir:55s} [{baseline_name:14s}] y_true={y_true}  "
                      f"inst_acc={inst_acc:.3f}  n_windows={len(probs)}  {agg_str}")


if __name__ == "__main__":
    main()
