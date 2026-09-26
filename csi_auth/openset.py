"""Open-set rejection: can we tell "this is neither Anjali nor Barath" instead of always forcing a
binary pick? Two approaches, tested honestly against strangers NOT used to pick any threshold:

1. Confidence-band rejection: reject if the identity model's own P(barath) stays near 0.5.
2. Distance-to-enrolled-centroid rejection: reject if a window's CNN+Attention embedding is far from
   BOTH Anjali's and Barath's centroid, in the model's actual learned embedding space (not the raw
   handcrafted features FINDINGS.md already found weren't well-separated by distance).

Strangers are split by PERSON into a calibration half (used only to pick thresholds) and a held-out
test half (used only to report how well those thresholds generalize) -- with ~11 distinct strangers
total this is a small, noisy split, treat the numbers as directional, not definitive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

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


def main():
    window_table, _, X_seq = build_dataset()
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


if __name__ == "__main__":
    main()
