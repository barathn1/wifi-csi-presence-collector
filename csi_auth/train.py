"""Auth (anjali/barath) vs non-auth (anyone else), evaluated leave-one-day-out across the 6 channel-6
walking days. Five models trained independently each fold -- SVM, RandomForest, HistGradientBoosting,
CNN+BiLSTM, CNN+self-attention -- plus a soft-vote ensemble of all five. Scored at both WINDOW
granularity (one decision per ~1s window) and SESSION granularity (average every test window's
probability within a session, then decide once) -- see session_aggregate(). Real labels only here --
see permutation_test.py for the label-shuffle chance control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, \
    precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset
from models import CnnAttention, CnnLstm, make_hgb, make_rf, make_svm, torch_model_predict_proba, \
    train_torch_model
from splits import assert_no_session_leakage, leave_one_day_out

RESULTS_PATH = "cache/auth_results.csv"


def inner_train_val_split(window_table: pd.DataFrame, train_idx: np.ndarray, val_frac: float = 0.2,
                           seed: int = 0):
    """Session-disjoint 80/20 split WITHIN the training days, used only to monitor CNN training --
    the held-out test day is never touched until final scoring."""
    sessions = window_table.iloc[train_idx]["session_dir"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sessions)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    val_sessions = set(shuffled[:n_val])
    is_val = window_table.iloc[train_idx]["session_dir"].isin(val_sessions).values
    return train_idx[~is_val], train_idx[is_val]


def metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }


def session_aggregate(proba: np.ndarray, sessions: np.ndarray, y_test: np.ndarray):
    """Average a model's per-window probability up to one number per test session (majority-vote's
    continuous cousin), then re-derive one true label per session. Window-to-window noise within a
    session should mostly average out -- this is the "decide from a few seconds, not one window" lever
    (matches the ARGUS-style segment-aggregation idea already used elsewhere in this project's history)."""
    df = pd.DataFrame({"session": sessions, "proba": proba, "y": y_test})
    per_session = df.groupby("session").agg(proba=("proba", "mean"), y=("y", "first"))
    return per_session["proba"].to_numpy(), per_session["y"].to_numpy()


def run(window_table: pd.DataFrame, X_stats: np.ndarray, X_seq: np.ndarray, y: np.ndarray,
        cnn_epochs: int = 8, label_source: str = "real", seed: int = 0) -> list[dict]:
    rows = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_table):
        assert_no_session_leakage(window_table, train_idx, test_idx)
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  {held_out_date}: skipped (single-class side)")
            continue

        # --- classical branch ---
        scaler = StandardScaler().fit(X_stats[train_idx])
        Xtr, Xte = scaler.transform(X_stats[train_idx]), scaler.transform(X_stats[test_idx])
        svm = make_svm().fit(Xtr, y_train)
        rf = make_rf().fit(Xtr, y_train)
        hgb = make_hgb().fit(Xtr, y_train)
        svm_proba = svm.predict_proba(Xte)[:, 1]
        rf_proba = rf.predict_proba(Xte)[:, 1]
        hgb_proba = hgb.predict_proba(Xte)[:, 1]

        # --- deep branch ---
        sub_mean = X_seq[train_idx].mean(axis=(0, 1), keepdims=True)
        sub_std = X_seq[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
        seq_norm = lambda X: ((X - sub_mean) / sub_std).astype(np.float32)
        inner_tr, inner_val = inner_train_val_split(window_table, train_idx, seed=seed)
        print(f"  {held_out_date}: training CNN+BiLSTM ({len(inner_tr)} train / {len(inner_val)} inner-val)...")
        lstm_model = train_torch_model(CnnLstm, seq_norm(X_seq[inner_tr]), y[inner_tr],
                                        seq_norm(X_seq[inner_val]), y[inner_val], epochs=cnn_epochs, seed=seed)
        cnn_proba = torch_model_predict_proba(lstm_model, seq_norm(X_seq[test_idx]))

        print(f"  {held_out_date}: training CNN+Attention...")
        attn_model = train_torch_model(CnnAttention, seq_norm(X_seq[inner_tr]), y[inner_tr],
                                        seq_norm(X_seq[inner_val]), y[inner_val], epochs=cnn_epochs, seed=seed)
        attn_proba = torch_model_predict_proba(attn_model, seq_norm(X_seq[test_idx]))

        ensemble_proba = (svm_proba + rf_proba + hgb_proba + cnn_proba + attn_proba) / 5.0
        test_sessions = window_table.iloc[test_idx]["session_dir"].to_numpy()

        for name, proba in [("svm", svm_proba), ("random_forest", rf_proba), ("hist_gradient_boost", hgb_proba),
                             ("cnn_bilstm", cnn_proba), ("cnn_attention", attn_proba),
                             ("ensemble", ensemble_proba)]:
            pred = (proba > 0.5).astype(int)
            m = metrics(y_test, pred)
            m.update(held_out_date=held_out_date, model=name, n_test=len(y_test), labels=label_source,
                      granularity="window")
            rows.append(m)

            sess_proba, sess_y = session_aggregate(proba, test_sessions, y_test)
            sess_pred = (sess_proba > 0.5).astype(int)
            sm = metrics(sess_y, sess_pred)
            sm.update(held_out_date=held_out_date, model=name, n_test=len(sess_y), labels=label_source,
                       granularity="session")
            rows.append(sm)

            print(f"    [{held_out_date}] {name}: window acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
                  f"| session acc={sm['accuracy']:.3f} bal_acc={sm['balanced_accuracy']:.3f} (n={len(sess_y)})")
    return rows


if __name__ == "__main__":
    window_table, X_stats, X_seq = build_dataset()
    y = window_table["auth"].to_numpy()
    rows = run(window_table, X_stats, X_seq, y)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n{len(df)} rows -> {RESULTS_PATH}")
    print(df.groupby(["model", "granularity"])[["accuracy", "balanced_accuracy", "f1"]].mean().round(3))
