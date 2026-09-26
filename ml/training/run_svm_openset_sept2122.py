"""Cross-day open-set identification on the 2026-09-21 / 2026-09-22 channel-6, 3-simultaneous-
receiver collection: auth (anjali, barath) vs. non_auth (every other person_id -- promoda/sumanth on
the 21st, vishwa/divya/harshitha on the 22nd, plus empty-room `none` sessions on both), starting with
an SVM baseline.

Preprocessing matches this project's existing BiLSTM channel-6 recipe (see `ablation_ch6_pipeline.py`'s
E9 experiment): Hampel filter (window=15, n_sigmas=3) -> Butterworth low-pass (order=5, cutoff=10Hz)
-> common-rate time resampling -> fixed 200-packet/50%-overlap windows
(`ml.data_pipeline.sept2122_ch6_openset_pipeline`). Since SVM (unlike the BiLSTM) doesn't consume a
raw packet sequence, each window is reduced to the same handcrafted per-subcarrier statistical
features (mean/std/skew/kurtosis over amplitude and phase, plus RSSI mean/std) already used by this
project's RandomForest/SVM baselines (`ml.data_pipeline.features`) -- unchanged from what those
already validated, just fed by the new preprocessing. Feature scaling (StandardScaler, fit on train
only) is added on top of that: the handcrafted features span very different magnitude ranges (raw
amplitude means vs. unit-scale skew/kurtosis), which the existing RandomForest baseline never needed
(tree splits are scale-invariant) but any linear/kernel SVM is sensitive to.

Uses `SGDClassifier(loss="hinge")` -- a linear SVM trained by stochastic gradient descent -- not
`ml.models.svm.make_svm`'s RBF-kernel `SVC`, and not `LinearSVC` either. Measured directly on this
dataset: the RBF `SVC`'s SMO solver took ~17 minutes PER FOLD (killed after 3 of 12 fits, 51 minutes
in); `LinearSVC`'s liblinear coordinate-descent solver then stalled for 2+ minutes on the very first
fold without converging, almost certainly because the 1026 handcrafted features are badly collinear
(mean/std/skew/kurtosis of ADJACENT subcarriers on a 20MHz channel are highly correlated with each
other) -- exactly the conditioning liblinear's coordinate descent handles poorly and SGD is robust to.
`svm.py`'s own docstring already flags the RBF kernel as expensive at ~24k rows; this dataset's
~28-35k windows/split pushed both classical solvers past practical. SGD fits the same data in
seconds and still optimizes the same hinge-loss (SVM) objective as `LinearSVC`, just via a different,
large-data-friendly solver.

Genuinely open-set already by construction, not just cross-day: the unauthorized identities on the
21st (promoda, sumanth) and the 22nd (vishwa, divya, harshitha) do not overlap at all, so training on
one day and testing on the other means every non-auth test person is a stranger the model never saw
during training -- true open-set generalization, not a closed-set proxy.

Three split conditions are reported, not just one:
1. Same-day: session-disjoint 5-fold CV within each date (upper bound -- same day/room/router).
2. Cross-day pairwise, both directions (the real question).
Unlike this project's neural-net models, `sklearn.svm.SVC` with `probability=False` has no
meaningful seed-to-seed variance (no random weight init/minibatch shuffling) -- a single fit per
split is enough, unlike the multi-seed averaging this repo's own BiLSTM/transformer trainers need.

    python -m ml.training.run_svm_openset_sept2122
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.features import build_feature_matrix
from ml.data_pipeline.sept2122_ch6_openset_pipeline import (
    TARGET_DATES,
    build_manifest,
    build_or_load_window_index,
    compute_dataset_target_rate_hz,
)
from ml.data_pipeline.splits import assert_no_group_leakage, stratified_session_disjoint_kfold
from ml.evaluation.metrics import compute_auroc, compute_eer

LOG_PATH = REPO_ROOT / "ml/evaluation/results/svm_openset_sept2122_log.csv"
LOG_FIELDS = [
    "timestamp", "split_type", "train", "test", "fold", "accuracy", "auroc", "eer",
    "far_unauthorized", "far_none", "frr", "recall_anjali", "recall_barath", "n_train", "n_test",
]


def build_dataset() -> tuple[np.ndarray, pd.DataFrame]:
    manifest = build_manifest()
    print(f"channel-6 (session,receiver) rows across {TARGET_DATES}: {len(manifest)}")
    print(manifest.groupby(["date", "label", "person_id"]).size())

    target_rate_hz = compute_dataset_target_rate_hz(manifest)
    window_index = build_or_load_window_index(manifest, target_rate_hz)
    print(f"total windows: {len(window_index)}")
    print(window_index.groupby(["date", "label"]).size())

    cache_x = REPO_ROOT / f"ml/data_pipeline/cache/sept2122_ch6_openset/features_{target_rate_hz:.2f}hz_n{len(window_index)}.npy"
    if cache_x.exists():
        X = np.load(cache_x)
    else:
        X = build_feature_matrix(window_index)
        cache_x.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_x, X)
    return X, window_index


def fit_and_score(X: np.ndarray, window_index: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                   seed: int) -> dict:
    y = window_index["is_auth"].values.astype(int)
    model = make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", alpha=1e-4, class_weight="balanced",
                                                            max_iter=1000, tol=1e-3, random_state=seed))
    model.fit(X[train_idx], y[train_idx])

    scores = model.decision_function(X[test_idx])  # higher = more "auth"-like
    pred = (scores > 0).astype(int)
    y_test = y[test_idx]
    raw_label_test = window_index["label"].values[test_idx]

    accuracy = float((pred == y_test).mean())
    auroc = compute_auroc(y_test, scores)
    eer, _ = compute_eer(y_test, scores)

    mask_unauth = raw_label_test == "unauthorized"
    far_unauthorized = float(pred[mask_unauth].mean()) if mask_unauth.any() else float("nan")
    mask_none = raw_label_test == "none"
    far_none = float(pred[mask_none].mean()) if mask_none.any() else float("nan")

    mask_auth = y_test == 1
    frr = float((pred[mask_auth] == 0).mean()) if mask_auth.any() else float("nan")

    def _recall(person: str) -> float:
        mask = window_index["person_id"].values[test_idx] == person
        return float((pred[mask] == 1).mean()) if mask.any() else float("nan")

    return {
        "accuracy": accuracy, "auroc": auroc, "eer": eer, "far_unauthorized": far_unauthorized,
        "far_none": far_none, "frr": frr, "recall_anjali": _recall("anjali"), "recall_barath": _recall("barath"),
        "n_train": len(train_idx), "n_test": len(test_idx),
    }


def same_day(X: np.ndarray, window_index: pd.DataFrame, seed: int, n_splits: int) -> list[dict]:
    print("\n=== SAME-DAY (session-disjoint k-fold within each date) ===")
    rows = []
    for date in TARGET_DATES:
        day_mask = (window_index["date"] == date).values
        day_idx = np.flatnonzero(day_mask)
        day_index_df = window_index.iloc[day_idx].reset_index(drop=True)
        n_sessions = day_index_df["session_dir"].nunique()
        splits = min(n_splits, n_sessions)
        fold_accs = []
        for fold, (tr, te) in enumerate(stratified_session_disjoint_kfold(day_index_df, n_splits=splits, seed=seed)):
            assert_no_group_leakage(day_index_df, tr, te, "session_dir")
            train_idx, test_idx = day_idx[tr], day_idx[te]
            m = fit_and_score(X, window_index, train_idx, test_idx, seed)
            fold_accs.append(m["accuracy"])
            print(f"  {date} fold {fold}: acc={m['accuracy']*100:.1f}% auroc={m['auroc']:.3f} "
                  f"eer={m['eer']:.3f} far_unauth={m['far_unauthorized']*100:.1f}% "
                  f"far_none={m['far_none']*100:.1f}% frr={m['frr']*100:.1f}% "
                  f"n_train={m['n_train']} n_test={m['n_test']}")
            rows.append({**m, "split_type": "same_day", "train": date, "test": date, "fold": fold})
        print(f"  {date} MEAN accuracy across {len(fold_accs)} folds: {np.mean(fold_accs)*100:.1f}%")
    return rows


def cross_day_pairwise(X: np.ndarray, window_index: pd.DataFrame, seed: int) -> list[dict]:
    print("\n=== CROSS-DAY (pairwise: train one date, test the other) ===")
    rows = []
    for train_date, test_date in [(TARGET_DATES[0], TARGET_DATES[1]), (TARGET_DATES[1], TARGET_DATES[0])]:
        train_idx = np.flatnonzero((window_index["date"] == train_date).values)
        test_idx = np.flatnonzero((window_index["date"] == test_date).values)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        m = fit_and_score(X, window_index, train_idx, test_idx, seed)
        print(f"  train={train_date} test={test_date}: acc={m['accuracy']*100:.1f}% auroc={m['auroc']:.3f} "
              f"eer={m['eer']:.3f} far_unauth={m['far_unauthorized']*100:.1f}% far_none={m['far_none']*100:.1f}% "
              f"frr={m['frr']*100:.1f}% recall_anjali={m['recall_anjali']*100:.1f}% "
              f"recall_barath={m['recall_barath']*100:.1f}% n_train={m['n_train']} n_test={m['n_test']}")
        rows.append({**m, "split_type": "cross_day_pairwise", "train": train_date, "test": test_date, "fold": 0})
    return rows


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = [{**{k: r.get(k, "") for k in LOG_FIELDS},
            "timestamp": datetime.now(timezone.utc).isoformat()} for r in rows]
    df = pd.DataFrame(out, columns=LOG_FIELDS)
    df.to_csv(LOG_PATH, mode="a" if LOG_PATH.exists() else "w", header=not LOG_PATH.exists(), index=False)


def main(seed: int, n_splits: int) -> None:
    X, window_index = build_dataset()

    all_rows = []
    all_rows += same_day(X, window_index, seed, n_splits)
    all_rows += cross_day_pairwise(X, window_index, seed)
    log_rows(all_rows)

    print("\n=== SUMMARY ===")
    print(f"{'split':<20}{'train':<14}{'test':<14}{'acc':>8}{'auroc':>8}{'eer':>8}{'far_unauth':>12}{'far_none':>10}")
    for r in all_rows:
        if r["split_type"] == "same_day" and r["fold"] != 0:
            continue
        label = r["split_type"] if r["split_type"] != "same_day" else "same_day (fold0)"
        print(f"{label:<20}{r['train']:<14}{r['test']:<14}{r['accuracy']*100:>7.1f}%{r['auroc']:>8.3f}"
              f"{r['eer']:>8.3f}{r['far_unauthorized']*100:>11.1f}%{r['far_none']*100:>9.1f}%")
    print(f"\nlogged {len(all_rows)} rows to {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()
    main(seed=args.seed, n_splits=args.n_splits)
