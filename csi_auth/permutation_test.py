"""Chance-level control: shuffle which SESSIONS are auth/non-auth (keeping the same session-level
auth/non-auth counts and all window features/labels-within-session intact), then repeat the exact same
leave-one-day-out evaluation. Real-label accuracy (train.py) must clearly beat this floor to mean
anything. Lighter settings than train.py (fewer windows/session, capped SVM iterations, fewer CNN
epochs, no ensemble) purely to keep total runtime bounded -- this is a chance-floor check, not the
headline result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset
from models import cnn_lstm_predict_proba, make_rf, make_svm, train_cnn_lstm
from splits import assert_no_session_leakage, leave_one_day_out
from train import inner_train_val_split, metrics

RESULTS_PATH = "cache/permutation_results.csv"
WINDOWS_PER_SESSION_SUBSAMPLE = 60
CNN_EPOCHS = 4
N_SHUFFLES = 5


def subsample_windows(window_table: pd.DataFrame, X_stats, X_seq, cap: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in window_table.groupby("session_dir"):
        idx = g.index.to_numpy()
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return window_table.loc[keep].reset_index(drop=True), X_stats[keep], X_seq[keep]


def permute_session_labels(window_table: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    true_labels = window_table.drop_duplicates("session_dir").set_index("session_dir")["auth"]
    shuffled = pd.Series(rng.permutation(true_labels.values), index=true_labels.index)
    return window_table["session_dir"].map(shuffled).to_numpy()


def run_one_shuffle(window_table, X_stats, X_seq, seed: int) -> list[dict]:
    y_shuf = permute_session_labels(window_table, seed)
    rows = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_table):
        assert_no_session_leakage(window_table, train_idx, test_idx)
        y_train, y_test = y_shuf[train_idx], y_shuf[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler().fit(X_stats[train_idx])
        Xtr, Xte = scaler.transform(X_stats[train_idx]), scaler.transform(X_stats[test_idx])
        svm = make_svm().fit(Xtr, y_train)
        rf = make_rf().fit(Xtr, y_train)
        svm_pred = svm.predict(Xte)
        rf_pred = rf.predict(Xte)

        sub_mean = X_seq[train_idx].mean(axis=(0, 1), keepdims=True)
        sub_std = X_seq[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
        seq_norm = lambda X: ((X - sub_mean) / sub_std).astype(np.float32)
        inner_tr, inner_val = inner_train_val_split(window_table, train_idx, seed=seed)
        model = train_cnn_lstm(seq_norm(X_seq[inner_tr]), y_shuf[inner_tr],
                                seq_norm(X_seq[inner_val]), y_shuf[inner_val], epochs=CNN_EPOCHS, seed=seed)
        cnn_pred = (cnn_lstm_predict_proba(model, seq_norm(X_seq[test_idx])) > 0.5).astype(int)

        for name, pred in [("svm", svm_pred), ("random_forest", rf_pred), ("cnn_lstm", cnn_pred)]:
            m = metrics(y_test, pred)
            m.update(held_out_date=held_out_date, model=name, n_test=len(y_test),
                      labels=f"shuffled_{seed}")
            rows.append(m)
        print(f"  shuffle {seed}, held out {held_out_date}: "
              f"svm={rows[-3]['accuracy']:.3f} rf={rows[-2]['accuracy']:.3f} cnn={rows[-1]['accuracy']:.3f}")
    return rows


if __name__ == "__main__":
    window_table, X_stats, X_seq = build_dataset()
    window_table, X_stats, X_seq = subsample_windows(window_table, X_stats, X_seq,
                                                       WINDOWS_PER_SESSION_SUBSAMPLE)
    print(f"permutation control on {len(window_table)} (subsampled) windows, {N_SHUFFLES} shuffles")

    all_rows = []
    for seed in range(N_SHUFFLES):
        all_rows += run_one_shuffle(window_table, X_stats, X_seq, seed)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n{len(df)} rows -> {RESULTS_PATH}")
    print(df.groupby("model")[["accuracy", "balanced_accuracy", "f1"]].agg(["mean", "std"]).round(3))
