"""Same two experiments as `run_walk_bilstm_loo_day.py` (cross-day identity accuracy) and
`run_walk_bilstm_stranger_openset.py` (auth-vs-unauthorized FAR/FRR), but fed the GLOBALLY-selected
[N, 400, 60] `walk_dataset` artifact (`build_walk_dataset_artifact.py`) instead of a per-fold
train-only subcarrier selection.

This exists purely as the requested apples-to-apples comparison against the per-fold numbers already
reported. It carries a real, deliberate leak the per-fold version was built to avoid: the top-30
subcarriers were chosen using variance computed over ALL 3 days pooled (including whichever day ends
up as the test fold), so the held-out day already influenced which features the model gets to see.
Any accuracy difference here vs. the per-fold numbers should be read as an upper bound inflated by
that leak, not as a fair preprocessing improvement.

    python -m ml.training.run_walk_bilstm_global_features --task identity
    python -m ml.training.run_walk_bilstm_global_features --task openset
    python -m ml.training.run_walk_bilstm_global_features --task both --epochs 20
"""
from __future__ import annotations

import argparse

import numpy as np

from ml.data_pipeline.build_walk_dataset_artifact import build_walk_dataset
from ml.data_pipeline.walk_bilstm_pipeline import (
    TARGET_DATES,
    apply_topk,
    build_dataset,
    build_unauthorized_walk_manifest,
    per_window_zscore,
)
from ml.training.run_walk_bilstm_loo_day import CLASSES as IDENTITY_CLASSES
from ml.training.run_walk_bilstm_loo_day import train_one_fold
from ml.training.run_walk_bilstm_stranger_openset import CLASSES as STRANGER_CLASSES
from ml.training.run_walk_bilstm_stranger_openset import UNAUTH_IDX, train_and_eval_fold


def run_identity(final: np.ndarray, meta, epochs: int, seeds: list[int]) -> None:
    print("\n=== [GLOBAL-FEATURE] LEAVE-ONE-DAY-OUT identity accuracy (anjali vs barath) ===")
    class_to_idx = {c: i for i, c in enumerate(IDENTITY_CLASSES)}
    fold_accs, fold_train_accs = [], []
    for held_out in TARGET_DATES:
        train_mask = (meta["date"] != held_out).values
        test_mask = (meta["date"] == held_out).values
        train_reduced, test_reduced = final[train_mask], final[test_mask]
        train_labels = meta.loc[train_mask, "person_id"].map(class_to_idx).values
        test_labels = meta.loc[test_mask, "person_id"].map(class_to_idx).values

        results = [train_one_fold(train_reduced, train_labels, test_reduced, test_labels, epochs, seed)
                   for seed in seeds]
        train_accs, test_accs, _models = zip(*results)
        mean_train, mean_test = float(np.mean(train_accs)), float(np.mean(test_accs))
        train_dates = [d for d in TARGET_DATES if d != held_out]
        print(f"  train={train_dates} test={held_out}: train_acc={mean_train*100:.1f}% test_acc={mean_test*100:.1f}%")
        fold_accs.append(mean_test)
        fold_train_accs.append(mean_train)
    print(f"\n[GLOBAL-FEATURE] mean train_acc={np.mean(fold_train_accs)*100:.1f}%  "
          f"mean test_acc across {len(fold_accs)} folds = {np.mean(fold_accs)*100:.1f}%")


def run_openset(final: np.ndarray, meta, idx: np.ndarray, epochs: int, seeds: list[int]) -> None:
    print("\n=== [GLOBAL-FEATURE] LEAVE-ONE-DAY-OUT + LEAVE-STRANGER(S)-OUT (3-way) ===")
    unauth_manifest = build_unauthorized_walk_manifest()
    unauth_windows_all, unauth_meta = build_dataset(unauth_manifest)  # [M, 400, 256], raw
    # same globally-fit idx (from the authorized-only artifact) + per-window zscore, applied to unauth
    unauth_final = per_window_zscore(apply_topk(unauth_windows_all, idx))

    class_to_idx = {c: i for i, c in enumerate(STRANGER_CLASSES)}
    fold_results = []
    for held_out in TARGET_DATES:
        auth_train_mask = (meta["date"] != held_out).values
        auth_test_mask = (meta["date"] == held_out).values
        auth_train_labels = meta.loc[auth_train_mask, "person_id"].map(class_to_idx).values
        auth_test_labels = meta.loc[auth_test_mask, "person_id"].map(class_to_idx).values

        held_out_people = set(unauth_meta.loc[(unauth_meta["date"] == held_out).values, "person_id"])
        unauth_train_mask = (unauth_meta["date"] != held_out).values & (~unauth_meta["person_id"].isin(held_out_people)).values
        unauth_test_mask = (unauth_meta["date"] == held_out).values

        train_reduced = np.concatenate([final[auth_train_mask], unauth_final[unauth_train_mask]], axis=0)
        train_labels = np.concatenate([auth_train_labels, np.full(unauth_train_mask.sum(), UNAUTH_IDX, np.int64)])
        auth_test_reduced, auth_test_lbls = final[auth_test_mask], auth_test_labels
        unauth_test_reduced = unauth_final[unauth_test_mask]

        seed_results = [train_and_eval_fold(train_reduced, train_labels, auth_test_reduced, auth_test_lbls,
                                             unauth_test_reduced, epochs, seed) for seed in seeds]
        mean = {k: float(np.mean([r[k] for r in seed_results])) for k in seed_results[0]}
        train_dates = [d for d in TARGET_DATES if d != held_out]
        print(f"  train={train_dates} test={held_out}: train_acc={mean['train_acc']*100:.1f}% "
              f"identity_acc={mean['identity_acc']*100:.1f}% FRR={mean['frr']*100:.1f}% "
              f"FAR(unauthorized accepted)={mean['far']*100:.1f}%")
        fold_results.append(mean)

    print(f"\n[GLOBAL-FEATURE] SUMMARY across {len(fold_results)} folds:")
    for k in ("train_acc", "identity_acc", "frr", "far"):
        print(f"  mean {k} = {np.mean([r[k] for r in fold_results])*100:.1f}%")


def main(task: str, epochs: int, seeds: list[int]) -> None:
    final, meta, idx = build_walk_dataset()
    print(f"walk_dataset (global-feature): {final.shape}")
    if task in ("identity", "both"):
        run_identity(final, meta, epochs, seeds)
    if task in ("openset", "both"):
        run_openset(final, meta, idx, epochs, seeds)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["identity", "openset", "both"], default="both")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()
    main(task=args.task, epochs=args.epochs, seeds=args.seeds)
