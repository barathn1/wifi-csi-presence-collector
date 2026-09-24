"""XGBoost on the same handcrafted feature vectors as svm_gait.py -- cheap sibling to the existing
RandomForest baseline (ml/models/baselines.py) and to the SVM variants. `feature_importances_` doubles
as the "which subcarriers/stats actually carry signal" diagnostic (OPEN_SET_MODEL_STRATEGY.md's
calibration section leans on exactly this kind of importance ranking).
"""
from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb


class GaitGBM:
    def __init__(self, n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.1,
                 use_gpu: bool | None = None):
        use_gpu = torch.cuda.is_available() if use_gpu is None else use_gpu
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            device="cuda" if use_gpu else "cpu", eval_metric="logloss",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaitGBM":
        self.model.fit(X, y)
        return self

    def genuine_score(self, X: np.ndarray) -> np.ndarray:
        """P(class 1) -- caller's y must use 1=authorized, 0=not, matching taskD's own convention."""
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        return proba[:, classes.index(1)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def top_features(self, feature_names: list[str], k: int = 20) -> list[tuple[str, float]]:
        importances = self.model.feature_importances_
        order = np.argsort(importances)[::-1][:k]
        return [(feature_names[i], float(importances[i])) for i in order]
