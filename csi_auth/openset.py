"""Open-set rejection: can we tell "this is neither Anjali nor Barath" instead of always forcing a
binary pick? Three approaches, all tested honestly against strangers NOT used to pick any
threshold/fit any model:

1. Confidence-band rejection: reject if the identity model's own P(barath) stays near 0.5.
2. Distance-to-enrolled-centroid rejection: reject if a window's CNN+Attention embedding is far from
   BOTH Anjali's and Barath's centroid, in the model's actual learned embedding space (not the raw
   handcrafted features FINDINGS.md already found weren't well-separated by distance).
3. One-class anomaly detection (OneClassSVM, IsolationForest): fit on Anjali+Barath pooled as one
   "known" class (doesn't need a labeled 3rd/stranger class the way a discriminative classifier would),
   flag low-likelihood windows as unknown. Tried on both the raw handcrafted stats features AND the
   CNN+Attention embedding, to see which representation the anomaly detector does better on.

Strangers are split by PERSON into a calibration half (used only to pick thresholds) and a held-out
test half (used only to report how well those thresholds generalize) -- with ~11 distinct strangers
total this is a small, noisy split, treat the numbers as directional, not definitive. The "known"
(Anjali/Barath) side of approach 3 also gets its own session-disjoint fit/eval split, so its
known-accept rate isn't just the model recognizing data it was fit on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from dataset import build_dataset
from models import CnnAttention, torch_model_embed, torch_model_predict_proba
from train_final import CHECKPOINT_DIR
import torch


def load_cnn_attention():
    ckpt = torch.load(CHECKPOINT_DIR / "cnn_attention_final.pt", weights_only=False)
    model = CnnAttention(n_subcarriers=ckpt["n_subcarriers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["sub_mean"], ckpt["sub_std"]


def split_strangers(window_table: pd.DataFrame, seed: int = 0):
    strangers = sorted(window_table.loc[~window_table["person_id"].isin(["anjali", "barath"]),
                                         "person_id"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(strangers)
    half = len(shuffled) // 2
    return set(shuffled[:half]), set(shuffled[half:])  # (calibration, held-out test)


def session_disjoint_split(window_table: pd.DataFrame, mask: np.ndarray, fit_frac: float = 0.7,
                            seed: int = 1):
    """Split the True-masked rows of window_table into a fit set and an eval set, by SESSION (not
    window) -- so the one-class model's "known-accept rate" is measured on sessions it never saw
    fit, the same leakage guard used everywhere else in this package."""
    sessions = window_table.loc[mask, "session_dir"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sessions)
    n_fit = int(round(len(shuffled) * fit_frac))
    fit_sessions = set(shuffled[:n_fit])
    is_fit = mask & window_table["session_dir"].isin(fit_sessions).values
    is_eval = mask & ~window_table["session_dir"].isin(fit_sessions).values
    return is_fit, is_eval


def evaluate_oneclass(name: str, X_fit, X_eval_known, X_calib_stranger, X_test_stranger) -> None:
    scaler = StandardScaler().fit(X_fit)
    Xf, Xk, Xc, Xt = (scaler.transform(a) for a in (X_fit, X_eval_known, X_calib_stranger, X_test_stranger))

    for clf_name, clf in [
        ("OneClassSVM", OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")),
        ("IsolationForest", IsolationForest(contamination=0.05, random_state=0, n_jobs=-1)),
    ]:
        clf.fit(Xf)
        score_known = clf.decision_function(Xk)   # higher = more "normal"/inlier, for both classes
        score_calib = clf.decision_function(Xc)
        score_test = clf.decision_function(Xt)

        y_calib = np.concatenate([np.ones(len(score_known)), np.zeros(len(score_calib))])
        s_calib = np.concatenate([score_known, score_calib])
        auc = roc_auc_score(y_calib, s_calib)

        threshold = np.percentile(score_known, 5)  # keep ~95% of held-out known windows
        known_accept = (score_known >= threshold).mean()
        stranger_reject = (score_test < threshold).mean()
        print(f"  [{name} / {clf_name}] calib AUC={auc:.3f}  known_accept={known_accept:.3f}  "
              f"held_out_stranger_reject={stranger_reject:.3f}")


def main():
    window_table, X_stats, X_seq = build_dataset()
    model, sub_mean, sub_std = load_cnn_attention()
    seq_norm = ((X_seq - sub_mean) / sub_std).astype(np.float32)

    print("extracting embeddings + P(barath) for every window...")
    proba = torch_model_predict_proba(model, seq_norm)
    embed = torch_model_embed(model, seq_norm)

    calib_strangers, test_strangers = split_strangers(window_table)
    print(f"strangers: {len(calib_strangers)} for calibration {sorted(calib_strangers)}, "
          f"{len(test_strangers)} held out for testing {sorted(test_strangers)}")

    is_anjali = (window_table["person_id"] == "anjali").values
    is_barath = (window_table["person_id"] == "barath").values
    is_calib_stranger = window_table["person_id"].isin(calib_strangers).values
    is_test_stranger = window_table["person_id"].isin(test_strangers).values

    # --- centroids from ALL anjali/barath windows (same data the deployed checkpoint trained on) ---
    anjali_centroid = embed[is_anjali].mean(axis=0)
    barath_centroid = embed[is_barath].mean(axis=0)

    def dist_to_nearest_centroid(e):
        return np.minimum(np.linalg.norm(e - anjali_centroid, axis=1),
                           np.linalg.norm(e - barath_centroid, axis=1))

    dist = dist_to_nearest_centroid(embed)
    conf_band = np.abs(proba - 0.5)  # low value = uncertain = near the decision boundary

    print("\n=== approach 1: confidence-band ===")
    print("AUC (does |P(barath)-0.5| separate known-person windows from calibration strangers?):")
    y_calib = np.concatenate([np.ones(is_anjali.sum() + is_barath.sum()), np.zeros(is_calib_stranger.sum())])
    score_calib = np.concatenate([conf_band[is_anjali | is_barath], conf_band[is_calib_stranger]])
    auc1 = roc_auc_score(y_calib, score_calib)
    print(f"  calibration AUC = {auc1:.3f} (0.5 = no separation, 1.0 = perfect)")
    fpr, tpr, thr = roc_curve(y_calib, score_calib)
    # pick the threshold that keeps 95% of known-person windows accepted
    best_idx = np.argmin(np.abs(tpr - 0.95))
    conf_threshold = thr[best_idx]
    print(f"  threshold picked on calibration set: accept if |P(barath)-0.5| > {conf_threshold:.3f} "
          f"(keeps ~95% of known-person windows)")

    print("\n=== approach 2: distance-to-centroid ===")
    y_calib2 = np.concatenate([np.ones(is_anjali.sum() + is_barath.sum()), np.zeros(is_calib_stranger.sum())])
    score_calib2 = np.concatenate([-dist[is_anjali | is_barath], -dist[is_calib_stranger]])  # closer = more "known"
    auc2 = roc_auc_score(y_calib2, score_calib2)
    print(f"  calibration AUC = {auc2:.3f} (0.5 = no separation, 1.0 = perfect)")
    fpr2, tpr2, thr2 = roc_curve(y_calib2, score_calib2)
    best_idx2 = np.argmin(np.abs(tpr2 - 0.95))
    dist_threshold = -thr2[best_idx2]
    print(f"  threshold picked on calibration set: accept if distance < {dist_threshold:.3f} "
          f"(keeps ~95% of known-person windows)")

    print("\n=== held-out evaluation (test strangers NEVER used above) ===")
    known_accept_rate_conf = (conf_band[is_anjali | is_barath] > conf_threshold).mean()
    stranger_reject_rate_conf = (conf_band[is_test_stranger] <= conf_threshold).mean()
    known_accept_rate_dist = (dist[is_anjali | is_barath] < dist_threshold).mean()
    stranger_reject_rate_dist = (dist[is_test_stranger] >= dist_threshold).mean()

    print(f"confidence-band: known-person accept rate = {known_accept_rate_conf:.3f}, "
          f"held-out-stranger reject rate = {stranger_reject_rate_conf:.3f}")
    print(f"distance-to-centroid: known-person accept rate = {known_accept_rate_dist:.3f}, "
          f"held-out-stranger reject rate = {stranger_reject_rate_dist:.3f}")

    # also report per-held-out-stranger breakdown, since n is small enough to just look at directly
    print("\nper-held-out-stranger reject rate (fraction of THEIR windows correctly flagged unknown):")
    for person in sorted(test_strangers):
        mask = window_table["person_id"].values == person
        print(f"  {person}: n={mask.sum():4d}  conf-band={( conf_band[mask] <= conf_threshold).mean():.3f}  "
              f"dist={(dist[mask] >= dist_threshold).mean():.3f}")

    print("\n=== approach 3: one-class anomaly detection (fit on Anjali+Barath pooled as one class) ===")
    is_known = is_anjali | is_barath
    is_fit, is_eval_known = session_disjoint_split(window_table, is_known)
    print(f"known-class fit/eval split: {is_fit.sum()} fit windows, {is_eval_known.sum()} held-out-session "
          f"eval windows")

    print("-- on raw handcrafted stats features (874-dim) --")
    evaluate_oneclass("stats", X_stats[is_fit], X_stats[is_eval_known],
                       X_stats[is_calib_stranger], X_stats[is_test_stranger])
    print("-- on CNN+Attention embedding (64-dim) --")
    evaluate_oneclass("embed", embed[is_fit], embed[is_eval_known],
                       embed[is_calib_stranger], embed[is_test_stranger])


if __name__ == "__main__":
    main()
