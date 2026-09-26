"""Train ONE final, deployable version of each of the three strongest models -- SVM, CNN+BiLSTM,
CNN+Attention -- on ALL 6 channel-6 walking days combined (Anjali vs Barath, no held-out test day).

This is deliberately different from every other script in this package: train.py/identity.py exist to
MEASURE cross-day generalization honestly (leave-one-day-out, nothing pooled), which is why there's no
single canonical "the" model anywhere else in this codebase -- each leave-one-day-out fold trains its
own model on a different 5/6 of the data and none of them is meant to be shipped. This script is the
one place that intentionally breaks that pattern, to produce an actual deployable artifact once the
leave-one-day-out results (see identity.py / identity_permutation.py output, and csi_auth/README.md)
already established that the underlying signal is real. Expect this model's OWN reported training-set
performance to look better than the honest cross-day numbers -- it has seen every day, so it is not a
fair estimate of live performance. Trust the cross-day numbers elsewhere in this repo for that; trust
this only as "the weights to actually run."
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset
from models import CnnAttention, CnnLstm, make_svm, torch_model_predict_proba, train_torch_model

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CNN_EPOCHS = 8
VAL_FRAC = 0.15


def session_disjoint_val_split(window_table, seed: int = 0):
    sessions = window_table["session_dir"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sessions)
    n_val = max(1, int(round(len(shuffled) * VAL_FRAC)))
    val_sessions = set(shuffled[:n_val])
    is_val = window_table["session_dir"].isin(val_sessions).values
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    window_table, X_stats, X_seq = build_dataset()
    mask = window_table["person_id"].isin(["anjali", "barath"]).values
    wt = window_table.loc[mask].reset_index(drop=True)
    Xs, Xq = X_stats[mask], X_seq[mask]
    y = (wt["person_id"] == "barath").to_numpy().astype(int)

    print(f"training final models on {len(wt)} windows, {wt['session_dir'].nunique()} sessions, "
          f"{wt['date'].nunique()} days (ALL days pooled -- no held-out day)")

    train_idx, val_idx = session_disjoint_val_split(wt)
    print(f"inner monitoring split: {len(train_idx)} train / {len(val_idx)} val windows "
          f"(session-disjoint, val used only to watch for -- not to select against -- the deep models)")

    # --- SVM (bundled with its scaler in one Pipeline so the .joblib is self-contained) ---
    svm_pipeline = make_pipeline(StandardScaler(), make_svm())
    svm_pipeline.fit(Xs, y)
    svm_path = CHECKPOINT_DIR / "svm_final.joblib"
    joblib.dump(svm_pipeline, svm_path)
    train_acc_svm = (svm_pipeline.predict(Xs) == y).mean()
    print(f"SVM: train-set accuracy {train_acc_svm:.3f} (NOT a cross-day estimate) -> {svm_path}")

    # --- deep branch: normalization stats saved alongside weights, needed at inference time ---
    sub_mean = Xq[train_idx].mean(axis=(0, 1), keepdims=True)
    sub_std = Xq[train_idx].std(axis=(0, 1), keepdims=True) + 1e-6
    seq_norm = lambda X: ((X - sub_mean) / sub_std).astype(np.float32)

    for name, model_cls in [("cnn_bilstm", CnnLstm), ("cnn_attention", CnnAttention)]:
        print(f"\ntraining final {name}...")
        model = train_torch_model(model_cls, seq_norm(Xq[train_idx]), y[train_idx],
                                   seq_norm(Xq[val_idx]), y[val_idx], epochs=CNN_EPOCHS)
        train_proba = torch_model_predict_proba(model, seq_norm(Xq))
        train_acc = ((train_proba > 0.5).astype(int) == y).mean()
        print(f"{name}: whole-pooled-set accuracy {train_acc:.3f} (NOT a cross-day estimate)")

        ckpt_path = CHECKPOINT_DIR / f"{name}_final.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "model_class": model_cls.__name__,
            "n_subcarriers": Xq.shape[2],
            "sub_mean": sub_mean, "sub_std": sub_std,
            "label_meaning": {0: "anjali", 1: "barath"},
        }, ckpt_path)
        print(f"  -> {ckpt_path}")

    meta = {
        "trained_on": "all 6 channel-6 walking days, Anjali vs Barath, pooled (no held-out day)",
        "n_windows": int(len(wt)), "n_sessions": int(wt["session_dir"].nunique()),
        "n_days": int(wt["date"].nunique()), "dates": sorted(wt["date"].unique().tolist()),
        "label_meaning": {"0": "anjali", "1": "barath"},
        "warning": "training-set/pooled-set accuracy above is NOT a cross-day generalization estimate "
                    "-- see csi_auth/README.md and identity_permutation.py output for the honest "
                    "leave-one-day-out numbers (82-90% session-level, confirmed above a permutation "
                    "chance floor) that justify shipping these weights at all.",
    }
    with open(CHECKPOINT_DIR / "README.md", "w") as f:
        f.write("# Final deployable checkpoints\n\n")
        f.write("```json\n" + json.dumps(meta, indent=2) + "\n```\n\n")
        f.write("Load with:\n\n```python\nimport torch, joblib\n"
                "svm = joblib.load('svm_final.joblib')  # scaler+SVM pipeline, call .predict_proba(X_stats)\n"
                "ckpt = torch.load('cnn_attention_final.pt')  # or cnn_bilstm_final.pt\n"
                "# reconstruct with models.CnnAttention(n_subcarriers=ckpt['n_subcarriers']), "
                "load_state_dict(ckpt['state_dict']), normalize inputs with ckpt['sub_mean']/['sub_std']\n"
                "```\n")
    print(f"\nwrote metadata -> {CHECKPOINT_DIR / 'README.md'}")


if __name__ == "__main__":
    main()
