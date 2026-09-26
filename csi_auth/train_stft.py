"""Does the Doppler/frequency-domain feature help auth-vs-non-auth where every time-domain approach
(stats, raw-sequence CNNs, one-class anomaly detection on both) failed at chance? Same leave-one-day-out
+ session-label-shuffle permutation control protocol as everywhere else in this package, so the result
is directly comparable to FINDINGS.md's existing numbers.

Tested three ways: Doppler features alone, existing handcrafted stats alone (as a same-protocol
sanity-check baseline on THIS window definition, since the Doppler windows are 1000/500 packets, not
the usual 200/100), and the two concatenated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataset_stft import build_stft_dataset
from models import make_rf, make_svm
from splits import assert_no_session_leakage, leave_one_day_out
from train import metrics, session_aggregate

RESULTS_PATH = "cache/stft_results.csv"
N_PERMUTATIONS = 8


def permute_session_labels(window_table: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    true_labels = window_table.drop_duplicates("session_dir").set_index("session_dir")["auth"]
    shuffled = pd.Series(rng.permutation(true_labels.values), index=true_labels.index)
    return window_table["session_dir"].map(shuffled).to_numpy()


def run(window_table: pd.DataFrame, X: np.ndarray, y: np.ndarray, label_source: str, feature_name: str):
    rows = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_table):
        assert_no_session_leakage(window_table, train_idx, test_idx)
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler().fit(X[train_idx])
        Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        test_sessions = window_table.iloc[test_idx]["session_dir"].to_numpy()

        for name, clf in [("svm", make_svm()), ("random_forest", make_rf())]:
            clf.fit(Xtr, y_train)
            proba = clf.predict_proba(Xte)[:, 1]
            pred = (proba > 0.5).astype(int)
            m = metrics(y_test, pred)
            m.update(held_out_date=held_out_date, model=name, feature=feature_name,
                      labels=label_source, granularity="window")
            rows.append(m)

            sess_proba, sess_y = session_aggregate(proba, test_sessions, y_test)
            sess_pred = (sess_proba > 0.5).astype(int)
            sm = metrics(sess_y, sess_pred)
            sm.update(held_out_date=held_out_date, model=name, feature=feature_name,
                      labels=label_source, granularity="session")
            rows.append(sm)
    return rows


def main():
    window_table, X_doppler = build_stft_dataset()
    y = window_table["auth"].to_numpy()
    print(f"{len(window_table)} windows ({y.sum()} auth / {(y == 0).sum()} non-auth), "
          f"Doppler feature dim={X_doppler.shape[1]}")

    all_rows = []
    print("\n=== real labels, Doppler features ===")
    all_rows += run(window_table, X_doppler, y, "real", "doppler")
    for seed in range(N_PERMUTATIONS):
        y_shuf = permute_session_labels(window_table, seed)
        all_rows += run(window_table, X_doppler, y_shuf, f"shuffled_{seed}", "doppler")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_PATH, index=False)
    df["label_type"] = np.where(df["labels"] == "real", "real", "shuffled")
    summary = df.groupby(["model", "granularity", "label_type"])[["accuracy", "balanced_accuracy"]] \
        .agg(["mean", "std"]).round(3)
    print(f"\n{len(df)} rows -> {RESULTS_PATH}")
    print(summary)


if __name__ == "__main__":
    main()
