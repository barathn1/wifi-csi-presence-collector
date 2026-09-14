"""SVM baseline -- same handcrafted per-window features as RandomForest (models/baselines.py), just a
different classical decision boundary. Included because it was named explicitly alongside RandomForest/
ResNet/transformer as a model to actually try, not because CSI amplitude/phase statistics have any
particular reason to be more linearly/RBF separable than tree-separable.

`probability=False`: sklearn's Platt-scaling probability calibration re-fits an internal 5-fold CV on
top of the RBF fit, which on ~24k rows (a typical Stage B fold) turns a several-second fit into a
many-minute one. `decision_function` (signed distance to the margin) is monotonic with confidence and
works fine as the ROC-based EER/AUROC score without paying for calibrated probabilities we don't need.
"""
from __future__ import annotations

from sklearn.svm import SVC


def make_svm(seed: int = 42) -> SVC:
    return SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, class_weight="balanced", random_state=seed)
