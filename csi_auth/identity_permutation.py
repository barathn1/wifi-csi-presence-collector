"""Chance-level control for the Anjali-vs-Barath identity task: shuffle which SESSIONS are anjali vs
barath (same counts, same window features), repeat leave-one-day-out. Mirrors permutation_test.py's
approach for the auth-vs-non-auth task, retargeted at identity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset
from identity import restrict_to_two_person
from models import cnn_lstm_predict_proba, make_rf, make_svm, train_cnn_lstm
from permutation_test import WINDOWS_PER_SESSION_SUBSAMPLE, CNN_EPOCHS, N_SHUFFLES, subsample_windows
from splits import assert_no_session_leakage, leave_one_day_out
from train import inner_train_val_split, metrics, session_aggregate

RESULTS_PATH = "cache/identity_permutation_results.csv"


def permute_person_labels(window_table: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    true_labels = window_table.drop_duplicates("session_dir").set_index("session_dir")["person_id"]
    shuffled = pd.Series(rng.permutation(true_labels.values), index=true_labels.index)
    y = window_table["session_dir"].map(shuffled)
    return (y == "barath").to_numpy().astype(int)


def run_one_shuffle(window_table, X_stats, X_seq, seed: int) -> list[dict]:
    y_shuf = permute_person_labels(window_table, seed)
    rows = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_table):
        assert_no_session_leakage(window_table, train_idx, test_idx)
        y_train, y_test = y_shuf[train_idx], y_shuf[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler().fit(X_stats[train_idx])
        Xtr, Xte = scaler.transform(X_stats[train_idx]), scaler.transform(X_stats[test_idx])
        svm_proba = make_svm().fit(Xtr, y_train).predict_proba(Xte)[:, 1]
        rf_proba = make_rf().fit(Xtr, y_train).predict_proba(Xte)[:, 1]

        sub_mean = X_seq[train_idx].mean(axis=(0, 1), keepdims=True)
        sub_std = X_seq[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
        seq_norm = lambda X: ((X - sub_mean) / sub_std).astype(np.float32)
        inner_tr, inner_val = inner_train_val_split(window_table, train_idx, seed=seed)
        model = train_cnn_lstm(seq_norm(X_seq[inner_tr]), y_shuf[inner_tr],
                                seq_norm(X_seq[inner_val]), y_shuf[inner_val], epochs=CNN_EPOCHS, seed=seed)
        cnn_proba = cnn_lstm_predict_proba(model, seq_norm(X_seq[test_idx]))
        test_sessions = window_table.iloc[test_idx]["session_dir"].to_numpy()

        fold_log = []
        for name, proba in [("svm", svm_proba), ("random_forest", rf_proba), ("cnn_lstm", cnn_proba)]:
            pred = (proba > 0.5).astype(int)
            m = metrics(y_test, pred)
            m.update(held_out_date=held_out_date, model=name, n_test=len(y_test),
                      labels=f"shuffled_{seed}", granularity="window")
            rows.append(m)

            sess_proba, sess_y = session_aggregate(proba, test_sessions, y_test)
            sess_pred = (sess_proba > 0.5).astype(int)
            sm = metrics(sess_y, sess_pred)
            sm.update(held_out_date=held_out_date, model=name, n_test=len(sess_y),
                      labels=f"shuffled_{seed}", granularity="session")
            rows.append(sm)
            fold_log.append(f"{name}: window={m['accuracy']:.3f} session={sm['accuracy']:.3f}")
        print(f"  shuffle {seed}, held out {held_out_date}: " + " | ".join(fold_log))
    return rows


if __name__ == "__main__":
    window_table, X_stats, X_seq = build_dataset()
    wt, Xs, Xq, _ = restrict_to_two_person(window_table, X_stats, X_seq)
    wt, Xs, Xq = subsample_windows(wt, Xs, Xq, WINDOWS_PER_SESSION_SUBSAMPLE)
    print(f"identity permutation control on {len(wt)} windows, {N_SHUFFLES} shuffles")

    all_rows = []
    for seed in range(N_SHUFFLES):
        all_rows += run_one_shuffle(wt, Xs, Xq, seed)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n{len(df)} rows -> {RESULTS_PATH}")
    print(df.groupby(["model", "granularity"])[["accuracy", "balanced_accuracy", "f1"]].agg(["mean", "std"]).round(3))
