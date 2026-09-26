"""Live-demo question: if Anjali/Barath just start walking, how many seconds in can the system commit
to a correct identity call? Trains each leave-one-day-out fold exactly like identity.py (using the
cached, subsampled TRAIN windows -- order doesn't matter for training), but for TEST sessions decodes
windows freshly IN CHRONOLOGICAL ORDER (no subsampling/shuffling), then simulates a live stream:
accumulate windows one at a time, and at checkpoints (10/30/45/60 windows seen so far) check whether
the running-mean-probability decision is correct yet. Also reports the actual elapsed wall-clock time
(from device_time_us) those checkpoints correspond to -- window count and real seconds are NOT the
same thing here, since capture rate varies session to session (confirmed ~200-490 Hz across this
dataset, not a fixed 1 window == 1 second).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cleaning import hampel_filter_amplitude
from data import build_session_table
from dataset import build_dataset
from decode import decode_dominant_bucket, load_npz
from features import window_features
from models import (CnnAttention, CnnLstm, make_hgb, make_rf, make_svm, torch_model_predict_proba,
                     train_torch_model)
from splits import leave_one_day_out
from subcarrier_mask import keep_mask
from train import inner_train_val_split
from windowing import STRIDE_PACKETS, WINDOW_PACKETS, make_windows

KEEP = keep_mask()
CHECKPOINTS = [10, 30, 45, 60, 90, 120, 150, 200, 300]
MAX_WINDOWS_FOR_DEMO = max(CHECKPOINTS)
RESULTS_PATH = "cache/demo_timing_results.csv"


def chronological_windows(npz_path: str, k_max: int = MAX_WINDOWS_FOR_DEMO):
    """First up to k_max windows of ONE session, in original time order. Returns (stats_feats,
    raw_seqs, elapsed_seconds_per_checkpoint_index)."""
    npz = load_npz(Path(npz_path))
    bucket = decode_dominant_bucket(npz)
    amp, phase, rssi = bucket["amplitude"], bucket["phase"], bucket["rssi"]
    amp, _ = hampel_filter_amplitude(amp)
    amp, phase = amp[:, KEEP], phase[:, KEEP]
    device_time_us = bucket["device_time_us"]

    stats, seqs, end_times = [], [], []
    for i, (start, end) in enumerate(make_windows(amp, phase, rssi, WINDOW_PACKETS, STRIDE_PACKETS)):
        if i >= k_max:
            break
        stats.append(window_features(amp[start:end], phase[start:end], rssi[start:end]))
        seqs.append(amp[start:end])
        end_times.append(device_time_us[end - 1])
    if not stats:
        return None
    t0 = device_time_us[0]
    elapsed_s = [(t - t0) / 1e6 for t in end_times]
    return np.stack(stats).astype(np.float32), np.stack(seqs).astype(np.float32), np.array(elapsed_s)


def main():
    sessions = build_session_table(motion="walking")
    sessions = sessions[sessions["person_id"].isin(["anjali", "barath"])].reset_index(drop=True)

    window_table, X_stats, X_seq = build_dataset()  # cached train-side features (masked+cleaned)
    id_mask = window_table["person_id"].isin(["anjali", "barath"]).values
    wt_id = window_table.loc[id_mask].reset_index(drop=True)
    Xs_id, Xq_id = X_stats[id_mask], X_seq[id_mask]

    rows = []
    for held_out_date in sorted(sessions["date"].unique()):
        train_mask = (wt_id["date"] != held_out_date).values
        test_sessions = sessions[sessions["date"] == held_out_date].reset_index(drop=True)
        if test_sessions["auth"].nunique() < 1 or test_sessions["person_id"].nunique() < 2:
            print(f"{held_out_date}: skipped (doesn't have both people)")
            continue

        train_idx = np.flatnonzero(train_mask)
        y_train = (wt_id.loc[train_mask, "person_id"] == "barath").to_numpy().astype(int)

        scaler = StandardScaler().fit(Xs_id[train_idx])
        Xtr = scaler.transform(Xs_id[train_idx])
        svm = make_svm().fit(Xtr, y_train)
        rf = make_rf().fit(Xtr, y_train)
        hgb = make_hgb().fit(Xtr, y_train)

        sub_mean = Xq_id[train_idx].mean(axis=(0, 1), keepdims=True)
        sub_std = Xq_id[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
        seq_norm = lambda X: ((X - sub_mean) / sub_std).astype(np.float32)
        inner_tr, inner_val = inner_train_val_split(wt_id.loc[train_mask].reset_index(drop=True),
                                                      np.arange(train_idx.size), seed=0)
        print(f"{held_out_date}: training deep models on {len(inner_tr)} windows...")
        lstm_model = train_torch_model(CnnLstm, seq_norm(Xq_id[train_idx][inner_tr]), y_train[inner_tr],
                                        seq_norm(Xq_id[train_idx][inner_val]), y_train[inner_val], epochs=8)
        attn_model = train_torch_model(CnnAttention, seq_norm(Xq_id[train_idx][inner_tr]), y_train[inner_tr],
                                        seq_norm(Xq_id[train_idx][inner_val]), y_train[inner_val], epochs=8)

        for _, srow in test_sessions.iterrows():
            result = chronological_windows(srow["npz_path"])
            if result is None:
                continue
            stats_seq, raw_seq, elapsed_s = result
            n = len(stats_seq)
            Xs_scaled = scaler.transform(stats_seq)
            Xq_normed = seq_norm(raw_seq)

            proba = {
                "svm": svm.predict_proba(Xs_scaled)[:, 1],
                "random_forest": rf.predict_proba(Xs_scaled)[:, 1],
                "hist_gradient_boost": hgb.predict_proba(Xs_scaled)[:, 1],
                "cnn_bilstm": torch_model_predict_proba(lstm_model, Xq_normed),
                "cnn_attention": torch_model_predict_proba(attn_model, Xq_normed),
            }
            proba["ensemble"] = np.mean(list(proba.values()), axis=0)
            y_true = int(srow["person_id"] == "barath")

            for k in CHECKPOINTS:
                if k > n:
                    continue
                for name, p in proba.items():
                    running_mean = p[:k].mean()
                    pred = int(running_mean > 0.5)
                    rows.append({
                        "held_out_date": held_out_date, "session_dir": srow["session_dir"],
                        "person_id": srow["person_id"], "model": name, "k_windows": k,
                        "elapsed_s": elapsed_s[k - 1], "correct": int(pred == y_true),
                    })
            print(f"  {srow['session_dir']} ({srow['person_id']}, {n} windows available, "
                  f"{elapsed_s[-1]:.1f}s covered)")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n{len(df)} rows -> {RESULTS_PATH}")
    summary = df.groupby(["model", "k_windows"]).agg(
        accuracy=("correct", "mean"), n_sessions=("correct", "size"),
        median_elapsed_s=("elapsed_s", "median")).round(3)
    print(summary)


if __name__ == "__main__":
    main()
