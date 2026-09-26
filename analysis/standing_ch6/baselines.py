"""Parts 4, 7, 8/9 -- quantitative identity test, permutation/label-shuffle control, and the two
"most important" generalization tests: leave-one-session-out and leave-one-day-out.

Target: anjali vs barath (the only two people with enough sessions/days to test cross-day generalization
at all -- see load_data.restricted_manifest / the manifest scan that motivated the channel-6-only,
standing-only scope). kNN / linear-kernel SVM / RandomForest on the standardized handcrafted feature
matrix, never a random packet-level split -- every split below groups by session_dir or date so
overlapping windows from one recording can never straddle train/test.

For every split type we run twice: once with the REAL person labels, once with labels permuted at the
SESSION level (same sessions, same features, but which session is "anjali" vs "barath" is shuffled) --
the real-label number must clear the shuffled-label number by a wide margin for the result to mean
anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analysis.standing_ch6.load_data import build
from ml.data_pipeline.splits import assert_no_group_leakage, leave_one_day_out, session_disjoint_kfold

RESULTS_DIR = Path(__file__).resolve().parent / "cache"
RESULTS_PATH = RESULTS_DIR / "baseline_results.csv"

MODELS = {
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    "svm_rbf": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0,
                                                      n_jobs=-1),
}


MAX_WINDOWS_PER_SESSION = 150  # cap so SVM fit time stays reasonable; windows overlap 50% anyway so this
# loses little independent information -- see raw_signal_plots.py's N_PACKETS_SHOWN discussion.


def subsample_per_session(wi: pd.DataFrame, X: np.ndarray, cap: int = MAX_WINDOWS_PER_SESSION, seed: int = 0):
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in wi.groupby("session_dir"):
        idx = g.index.to_numpy()
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    keep = np.sort(np.concatenate(keep))
    return wi.loc[keep].reset_index(drop=True), X[keep]


def restrict_to_two_person(window_index: pd.DataFrame, X: np.ndarray):
    mask = window_index["identity"].isin(["anjali", "barath"]).values
    wi = window_index.loc[mask].reset_index(drop=True)
    X = X[mask]
    wi, X = subsample_per_session(wi, X)
    y = (wi["identity"] == "barath").to_numpy(dtype=int)
    return wi, X, y


def permute_session_labels(window_index: pd.DataFrame, seed: int) -> np.ndarray:
    """Same sessions, same feature rows -- reassign each session_dir's label by shuffling the
    session-level (not window-level) mapping, so window-overlap/session artifacts stay intact and only
    the person<->session correspondence is randomized."""
    rng = np.random.default_rng(seed)
    sessions = window_index["session_dir"].unique()
    true_labels = window_index.drop_duplicates("session_dir").set_index("session_dir")["identity"]
    shuffled_labels = pd.Series(rng.permutation(true_labels.values), index=true_labels.index)
    y_shuffled = window_index["session_dir"].map(shuffled_labels)
    return (y_shuffled == "barath").to_numpy(dtype=int)


def evaluate_fold(model_fn, X_train, y_train, X_test, y_test) -> dict:
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    clf = model_fn()
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "n_test": len(y_test),
        "confusion_matrix": cm.tolist(),
    }


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "n_test": len(y_true),
        "confusion_matrix": cm.tolist(),
    }


def run_split_scheme_pooled(window_index, X, y, scheme_name: str, splits, group_col: str,
                             label_name: str = "real") -> list[dict]:
    """For LOSO-style schemes where a single fold's test side is necessarily single-class (holding out
    one of only two people's one session), per-fold balanced_accuracy/F1 are degenerate. Instead, collect
    every fold's (y_true, y_pred) and score ONCE over the pooled predictions -- standard practice for
    leave-one-group-out with few groups. Still requires every fold's TRAIN side to have both classes."""
    rows = []
    for model_name, model_fn in MODELS.items():
        y_true_all, y_pred_all = [], []
        n_folds_used = 0
        for fold_name, train_idx, test_idx in splits:
            assert_no_group_leakage(window_index, train_idx, test_idx, group_col)
            if len(np.unique(y[train_idx])) < 2:
                continue
            scaler = StandardScaler().fit(X[train_idx])
            clf = model_fn()
            clf.fit(scaler.transform(X[train_idx]), y[train_idx])
            pred = clf.predict(scaler.transform(X[test_idx]))
            y_true_all.append(y[test_idx])
            y_pred_all.append(pred)
            n_folds_used += 1
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        res = pooled_metrics(y_true_all, y_pred_all)
        res.update(scheme=scheme_name, fold=f"pooled_{n_folds_used}folds", model=model_name,
                   labels=label_name)
        rows.append(res)
    return rows


def run_split_scheme(window_index, X, y, scheme_name: str, splits, group_col: str) -> list[dict]:
    rows = []
    for fold_name, train_idx, test_idx in splits:
        assert_no_group_leakage(window_index, train_idx, test_idx, group_col)
        # both classes must be present on both sides, else metrics are meaningless
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            print(f"  [{scheme_name}] fold {fold_name}: skipped (single-class side)")
            continue
        for model_name, model_fn in MODELS.items():
            real = evaluate_fold(model_fn, X[train_idx], y[train_idx], X[test_idx], y[test_idx])
            real.update(scheme=scheme_name, fold=str(fold_name), model=model_name, labels="real")
            rows.append(real)
    return rows


def run_permutation_control(window_index, X, scheme_name: str, splits_fn, group_col: str,
                             n_permutations: int = 10) -> list[dict]:
    rows = []
    for seed in range(n_permutations):
        y_shuf = permute_session_labels(window_index, seed)
        for fold_name, train_idx, test_idx in splits_fn():
            if len(np.unique(y_shuf[train_idx])) < 2 or len(np.unique(y_shuf[test_idx])) < 2:
                continue
            for model_name, model_fn in MODELS.items():
                res = evaluate_fold(model_fn, X[train_idx], y_shuf[train_idx], X[test_idx], y_shuf[test_idx])
                res.update(scheme=scheme_name, fold=str(fold_name), model=model_name,
                           labels=f"shuffled_{seed}")
                rows.append(res)
    return rows


def main():
    window_index, X = build()
    wi, X2, y = restrict_to_two_person(window_index, X)
    print(f"anjali vs barath: {len(wi)} windows, {wi['session_dir'].nunique()} sessions, "
          f"{wi['date'].nunique()} days")
    print(wi.groupby(["date", "identity"])["session_dir"].nunique())

    all_rows = []

    def _named_kfold():
        return [(i, tr, te) for i, (tr, te) in enumerate(session_disjoint_kfold(wi, n_splits=5))]

    print("\n=== cross-session (5-fold, GroupKFold on session_dir) ===")
    all_rows += run_split_scheme(wi, X2, y, "cross_session_5fold", _named_kfold(), "session_dir")
    all_rows += run_permutation_control(wi, X2, "cross_session_5fold", _named_kfold, "session_dir")

    print("=== leave-one-session-out (LeaveOneGroupOut on session_dir, POOLED across folds) ===")
    logo = LeaveOneGroupOut()
    groups = wi["session_dir"].values
    session_splits = [(groups[test_idx][0], train_idx, test_idx)
                       for train_idx, test_idx in logo.split(X2, y, groups=groups)]
    all_rows += run_split_scheme_pooled(wi, X2, y, "leave_one_session_out", session_splits, "session_dir")
    n_loso_perms = 10
    for seed in range(n_loso_perms):
        y_shuf = permute_session_labels(wi, seed)
        all_rows += run_split_scheme_pooled(wi, X2, y_shuf, "leave_one_session_out", session_splits,
                                             "session_dir", label_name=f"shuffled_{seed}")

    print("=== leave-one-day-out (the key cross-day generalization test) ===")
    day_splits = list(leave_one_day_out(wi))
    all_rows += run_split_scheme(wi, X2, y, "leave_one_day_out", day_splits, "session_dir")
    all_rows += run_permutation_control(
        wi, X2, "leave_one_day_out", lambda: list(leave_one_day_out(wi)), "session_dir")

    df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n{len(df)} fold/model results -> {RESULTS_PATH}")

    df["label_type"] = np.where(df["labels"] == "real", "real", "shuffled")
    summary = (df.groupby(["scheme", "model", "label_type"])[["accuracy", "balanced_accuracy", "f1_macro"]]
               .agg(["mean", "std"]).round(3))
    print(summary)


if __name__ == "__main__":
    main()
