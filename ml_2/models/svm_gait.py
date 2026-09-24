"""SVM-based gait models on the handcrafted per-subcarrier feature vectors
(ml.data_pipeline.features.build_feature_matrix) -- priority model family per this round's explicit
ask, following Wii (Sensors 2017): GMM for stranger rejection + SVM for identity among knowns.

Three variants, all operating on the SAME StandardScaler-normalized feature vectors:
  1. OneClassGaitSVM  -- RBF one-class SVM fit on authorized-only features. Open-set reject gate: the
     direct analog of Wii's GMM stage, but an SVM (per this round's specific ask) rather than a GMM.
  2. BinaryGaitSVM    -- ordinary RBF SVC (balanced class weight), authorized vs everything else.
     Closed-set sibling to the existing RandomForest baseline -- same task, same features, different
     classifier, so the SVM-vs-RandomForest comparison is controlled.
  3. GaitGmmSvmTwoStage -- the literal Wii replica: GMM stranger-reject stage 1, multiclass SVC
     identity stage 2 (only reached for inputs stage 1 accepts as "known").

All three expose a `genuine_score(X) -> higher-is-more-authorized-like` array so
ml.evaluation.metrics.compute_eer/compute_auroc can score them exactly like the deep models.
"""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM


class OneClassGaitSVM:
    def __init__(self, nu: float = 0.1, gamma: str | float = "scale"):
        self.scaler = StandardScaler()
        self.svm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

    def fit(self, X_authorized: np.ndarray) -> "OneClassGaitSVM":
        Xs = self.scaler.fit_transform(X_authorized)
        self.svm.fit(Xs)
        return self

    def genuine_score(self, X: np.ndarray) -> np.ndarray:
        """decision_function: positive = inlier/authorized-like, negative = outlier -- already the
        higher-is-more-genuine convention compute_eer/compute_auroc expect, no sign flip needed."""
        return self.svm.decision_function(self.scaler.transform(X))

    def predict_authorized(self, X: np.ndarray) -> np.ndarray:
        return (self.svm.predict(self.scaler.transform(X)) == 1).astype(int)


class BinaryGaitSVM:
    def __init__(self, C: float = 1.0, gamma: str | float = "scale"):
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", probability=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryGaitSVM":
        Xs = self.scaler.fit_transform(X)
        self.svm.fit(Xs, y)
        return self

    def genuine_score(self, X: np.ndarray) -> np.ndarray:
        """P(class 1) -- caller's y must use 1=authorized, 0=not, matching taskD's own convention."""
        proba = self.svm.predict_proba(self.scaler.transform(X))
        classes = list(self.svm.classes_)
        return proba[:, classes.index(1)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.svm.predict(self.scaler.transform(X))


class GaitGmmSvmTwoStage:
    """Wii replica. Stage 1 (GMM) rejects strangers using ONLY authorized-user features -- no intruder
    data needed to fit it, same "no prior knowledge of intruders" property CAUTION's threshold has.
    Stage 2 (multiclass SVC) predicts identity among whichever known users stage 1 accepted the sample
    as belonging to -- trained on taskB-style identity labels (e.g. anjali vs barath)."""

    def __init__(self, n_gmm_components: int = 3, gmm_reject_percentile: float = 5.0, svc_C: float = 1.0):
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_gmm_components, random_state=0)
        self.gmm_reject_percentile = gmm_reject_percentile
        self.gmm_threshold_: float | None = None
        self.svc = SVC(kernel="rbf", C=svc_C, gamma="scale", probability=True)
        self.has_identity_stage_ = False

    def fit(self, X_authorized: np.ndarray, X_identity: np.ndarray, y_identity: np.ndarray) -> "GaitGmmSvmTwoStage":
        """`X_authorized`: every authorized-window feature vector, used to fit the GMM AND to set its
        reject threshold at the `gmm_reject_percentile`-th percentile of authorized-only log-likelihood
        (so ~gmm_reject_percentile% of genuine authorized windows are themselves borderline-rejected --
        the same "calibrate the threshold using only known-class data" principle as CAUTION's threshold
        and the OC-SVM's nu parameter, just expressed as a percentile here instead of nu).
        `X_identity`/`y_identity`: authorized-only features + identity labels for the stage-2 SVC --
        skipped (has_identity_stage_ stays False) when fewer than 2 identities are present in this
        fold's training data (e.g. a leave-one-day-out fold with only one authorized person recorded
        that day), since SVC needs >=2 classes and stage 1's reject decision doesn't depend on it."""
        Xs = self.scaler.fit_transform(X_authorized)
        self.gmm.fit(Xs)
        log_lik = self.gmm.score_samples(Xs)
        self.gmm_threshold_ = float(np.percentile(log_lik, self.gmm_reject_percentile))

        if len(np.unique(y_identity)) >= 2:
            Xi = self.scaler.transform(X_identity)
            self.svc.fit(Xi, y_identity)
            self.has_identity_stage_ = True
        return self

    def genuine_score(self, X: np.ndarray) -> np.ndarray:
        """GMM log-likelihood IS the genuine score (higher = more authorized-gait-like) -- exactly what
        the open-set reject decision is made from, before stage 2 ever runs."""
        return self.gmm.score_samples(self.scaler.transform(X))

    def predict_authorized(self, X: np.ndarray) -> np.ndarray:
        return (self.genuine_score(X) >= self.gmm_threshold_).astype(int)

    def predict_identity(self, X: np.ndarray) -> np.ndarray:
        """Only meaningful for inputs predict_authorized already accepted -- caller's responsibility to
        filter, same two-stage discipline Wii uses (identity is never asked of a rejected stranger)."""
        return self.svc.predict(self.scaler.transform(X))
