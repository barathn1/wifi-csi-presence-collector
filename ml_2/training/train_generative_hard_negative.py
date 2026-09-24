"""Trains a small VAE on authorized-only features, samples synthetic near-boundary "hard negative"
feature vectors, and uses them two ways: (1) as an additional DIAGNOSTIC stress-test of the OC-SVM
reject gate (ml_2/models/svm_gait.py) -- report what fraction of synthetic hard negatives the OC-SVM
would wrongly accept, alongside its real false-accept rate against genuine unauthorized/none data; and
(2) folded into OC-SVM training as extra negative-side signal for a class-weighted SVC variant, to see
whether training WITH synthetic hard negatives improves real open-set rejection.

Synthetic negatives are never substituted for real evaluation data -- the real open-set/cross-day
numbers always come from genuine held-out unauthorized/none windows.

    python3 -m ml_2.training.train_generative_hard_negative
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_2.data.decode import REPO_ROOT
from ml_2.data.metrics import compute_auroc, compute_eer
from ml_2.data.splits import assert_no_group_leakage, leave_one_unauthorized_person_out
from ml_2.models.generative_hard_negative import sample_hard_negatives, train_vae
from ml_2.models.svm_gait import OneClassGaitSVM
from ml_2.training.common_data import Channel6Dataset, feature_matrix_for, load_channel6_dataset
from ml_2.training.gpu_utils import DEVICE

LOG_PATH = REPO_ROOT / "ml_2/evaluation/results/generative_hard_negative_log.csv"
SESSION_CSV_PATH = REPO_ROOT / "ml_2/evaluation/results/session_breakdown.csv"
LOG_FIELDNAMES = ["timestamp", "model", "held_out", "accuracy", "eer", "auroc",
                   "session_accuracy", "session_auroc", "n_sessions",
                   "false_accept_unauthorized", "false_accept_none", "synthetic_hard_negative_accept_rate",
                   "n_train", "n_test", "notes"]
LATENT_DIM = 16
MAX_SVM_TRAIN_SAMPLES = 6000  # see train_svm_gait.py -- RBF SVC/OneClassSVM don't scale to this
# dataset's 100k+-sample folds; the VAE itself (a small MLP) is fine on the full authorized set.


def _subsample(indices: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=cap, replace=False)


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def main() -> None:
    print(f"training on device: {DEVICE}")
    dataset: Channel6Dataset = load_channel6_dataset()
    window_index = dataset.window_index
    X = feature_matrix_for(window_index, dataset)

    print("\n=== VAE-augmented OC-SVM open-set: leave-one-unauthorized-person-out ===")
    fold_metrics = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(window_index, include_none=True):
        assert_no_group_leakage(window_index, train_idx, test_idx)
        train_rows, test_rows = window_index.iloc[train_idx], window_index.iloc[test_idx]
        y_train_auth = (train_rows["label"] == "authorized").values
        X_train_auth = X[train_idx][y_train_auth]

        vae = train_vae(X_train_auth, latent_dim=LATENT_DIM, device=DEVICE)  # small MLP, full batch is fine

        # SVC/OneClassSVM don't scale to this many rows -- subsample only for the kernel-SVM fits below,
        # not for the VAE above (which is cheap on the full authorized set regardless of size).
        svm_fit_rows = _subsample(np.arange(len(X_train_auth)), MAX_SVM_TRAIN_SAMPLES)
        X_svm_fit = X_train_auth[svm_fit_rows]
        hard_neg = sample_hard_negatives(vae, n_samples=len(X_svm_fit), latent_dim=LATENT_DIM, device=DEVICE)

        # baseline: plain OC-SVM (no synthetic negatives)
        plain = OneClassGaitSVM(nu=0.1).fit(X_svm_fit)
        # augmented: binary SVC trained on real-authorized (+1) vs synthetic-hard-negative (-1)
        scaler = StandardScaler().fit(np.concatenate([X_svm_fit, hard_neg]))
        svc = SVC(kernel="rbf", class_weight="balanced", probability=True)
        svc.fit(scaler.transform(np.concatenate([X_svm_fit, hard_neg])),
                np.concatenate([np.ones(len(X_svm_fit)), np.zeros(len(hard_neg))]))

        X_test = X[test_idx]
        y_test = (test_rows["label"] == "authorized").astype(int).values
        hard_neg_test = sample_hard_negatives(vae, n_samples=200, latent_dim=LATENT_DIM, device=DEVICE)

        for model_name, score in [
            ("oc_svm_plain", plain.genuine_score(X_test)),
            ("svc_vae_augmented", svc.predict_proba(scaler.transform(X_test))[:, list(svc.classes_).index(1.0)]),
        ]:
            eer, threshold = compute_eer(y_test, score)
            auroc = compute_auroc(y_test, score)
            pred = (score >= threshold).astype(int) if not np.isnan(threshold) else (score >= 0).astype(int)
            hard_neg_score = plain.genuine_score(hard_neg_test) if model_name == "oc_svm_plain" \
                else svc.predict_proba(scaler.transform(hard_neg_test))[:, list(svc.classes_).index(1.0)]
            hard_neg_accept_rate = float((hard_neg_score >= (threshold if not np.isnan(threshold) else 0)).mean())
            m = {"model": model_name, "held_out": held_out, "accuracy": float((pred == y_test).mean()),
                 "eer": eer, "auroc": auroc, "synthetic_hard_negative_accept_rate": hard_neg_accept_rate,
                 "n_train": len(train_idx), "n_test": len(test_idx)}
            for neg in ("unauthorized", "none"):
                mask = test_rows["label"].values == neg
                if mask.sum() > 0:
                    m[f"false_accept_{neg}"] = float((pred[mask] == 1).mean())

            from ml_2.data.metrics import format_session_breakdown, save_session_breakdown_csv, session_level_metrics
            sm = session_level_metrics(y_test, score, pred, test_rows["session_dir"].values,
                                        test_rows["date"].values)
            m["session_accuracy"], m["session_auroc"], m["n_sessions"] = \
                sm["session_accuracy"], sm["session_auroc"], sm["n_sessions"]
            m["_session_breakdown_str"] = format_session_breakdown(sm)
            save_session_breakdown_csv(sm["per_session_table"], SESSION_CSV_PATH, model_name,
                                        "open_set_loo_stranger", held_out)

            fold_metrics.append(m)
            log_rows([{**m, "timestamp": datetime.now(timezone.utc).isoformat(),
                        "notes": "leave-one-unauthorized-person-out"}])

        print(f"  held out '{held_out}': "
              f"plain WINDOW auroc={fold_metrics[-2]['auroc']:.3f} SESSION auroc={fold_metrics[-2]['session_auroc']:.3f} "
              f"fa_unauth={fold_metrics[-2].get('false_accept_unauthorized', float('nan')):.3f} | "
              f"vae-augmented WINDOW auroc={fold_metrics[-1]['auroc']:.3f} SESSION auroc={fold_metrics[-1]['session_auroc']:.3f} "
              f"fa_unauth={fold_metrics[-1].get('false_accept_unauthorized', float('nan')):.3f} "
              f"hard_neg_accept={fold_metrics[-1]['synthetic_hard_negative_accept_rate']:.3f}")
        print(fold_metrics[-2].pop("_session_breakdown_str", ""))
        print(fold_metrics[-1].pop("_session_breakdown_str", ""))

    for name in ("oc_svm_plain", "svc_vae_augmented"):
        rows = [m for m in fold_metrics if m["model"] == name]
        print(f"  MEAN[{name}] window_auroc={np.nanmean([m['auroc'] for m in rows]):.3f} "
              f"session_auroc={np.nanmean([m['session_auroc'] for m in rows]):.3f} "
              f"false_accept_unauth={np.nanmean([m.get('false_accept_unauthorized', np.nan) for m in rows]):.3f}")
    print(f"\nDone. Results -> {LOG_PATH}")


if __name__ == "__main__":
    main()
