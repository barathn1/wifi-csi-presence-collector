"""Metrics shared by every model in the zoo: accuracy/confusion matrix for classifiers, EER/AUROC for
verification scores. EER is the standard threshold-tuning metric for CSI biometrics per the SoK survey
cited in RESEARCH_NOTES.md (arXiv:2511.11381), which recommends it stay under 5% for security-grade use.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """y_true: 1=genuine/authorized, 0=impostor/unauthorized. scores: higher = more likely genuine.
    Returns (eer, threshold_at_eer). (nan, nan) if the fold has only one class -- e.g. a session-disjoint
    split that happens to hold out zero authorized sessions -- since FPR/FNR aren't defined there."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thresholds[idx])


def compute_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def fold_summary(fold_metrics: list[dict]) -> dict:
    """Mean +/- std across folds for every numeric key present in all fold dicts."""
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics if not np.isnan(m[k])])
        summary[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        summary[f"{k}_std"] = float(vals.std()) if len(vals) else float("nan")
    return summary


def labeled_confusion_matrix(y_true, y_pred, labels=None) -> tuple[np.ndarray, list]:
    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels
