"""Classical-ML baseline (Stage 1 of the combination sweep): RandomForest on the handcrafted per-window
statistics from data_pipeline/features.py. Cheap, interpretable (feature_importances_ feeds
visualization/feature_importance.py), and a floor every deep model in models/ should beat.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def make_baseline(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed, class_weight="balanced")


def evaluate_folds(X: np.ndarray, y: np.ndarray, splits, model_factory=make_baseline) -> list[dict]:
    """splits: iterable of (train_idx, test_idx) row-position arrays into X/y."""
    fold_metrics = []
    for train_idx, test_idx in splits:
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], pred)
        fold_metrics.append({"accuracy": acc, "n_train": len(train_idx), "n_test": len(test_idx)})
    return fold_metrics


def evaluate_binary_scores(X: np.ndarray, y: np.ndarray, splits, model_factory=make_baseline) -> list[dict]:
    """For tasks where we also want an EER/AUROC-style score (binary y only): uses predict_proba."""
    from ml.evaluation.metrics import compute_auroc, compute_eer

    fold_metrics = []
    for train_idx, test_idx in splits:
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        acc = accuracy_score(y[test_idx], pred)
        eer, thresh = compute_eer(y[test_idx], proba)
        auroc = compute_auroc(y[test_idx], proba)
        fold_metrics.append({
            "accuracy": acc, "eer": eer, "auroc": auroc,
            "n_train": len(train_idx), "n_test": len(test_idx),
        })
    return fold_metrics
