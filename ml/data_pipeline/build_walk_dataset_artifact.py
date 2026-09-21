"""Materialize the final `walk_dataset` artifact the original spec's goal described: a single
[N_windows, 400, 60] clean tensor (top-30-variance amplitude subcarriers + their matching phase
subcarriers, per-window per-channel normalized), for the walking-only, channel-6, 2026-09-15/16/17
authorized data (anjali/barath).

Difference from the per-fold reduction used in `ml/training/run_walk_bilstm_loo_day.py`: THIS artifact
selects its top-30 subcarriers ONCE, globally, over the whole pooled 3-day dataset -- matching the
pasted spec's `preprocess_day` literally (subcarrier selection and normalization happen once, over
"the dataset", not per train/test split). That makes it a convenient single reusable tensor to inspect
or feed into something other than the day-disjoint CV loop, but it is NOT what cross-day model
training should use directly: fitting subcarrier selection over data that includes the day you'll
later evaluate on is a (mild) cross-day leak. The LODO training script re-derives its own
train-fold-only selection for exactly that reason -- this script is for producing/inspecting the
artifact itself, not for training.

    python -m ml.data_pipeline.build_walk_dataset_artifact
"""
from __future__ import annotations

import numpy as np

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.walk_bilstm_pipeline import (
    TOP_K_SUBCARRIERS,
    apply_topk,
    build_dataset,
    build_walk_manifest,
    per_window_zscore,
    select_topk_variance,
)

ARTIFACT_PATH = REPO_ROOT / "ml/data_pipeline/cache/walk_bilstm/walk_dataset_400x60.npz"


def build_walk_dataset() -> tuple[np.ndarray, "pd.DataFrame", np.ndarray]:
    """Returns (final [N, 400, 60] tensor, per-window metadata DataFrame, selected subcarrier indices
    into the original 128)."""
    manifest = build_walk_manifest()
    windows_all, meta = build_dataset(manifest)  # [N, 400, 256]: 128 amp + 128 phase, pre-selection

    idx = select_topk_variance(windows_all, k=TOP_K_SUBCARRIERS)
    reduced = apply_topk(windows_all, idx)          # [N, 400, 60]: 30 amp + 30 phase
    final = per_window_zscore(reduced)              # per-window, per-channel z-score
    return final, meta, idx


if __name__ == "__main__":
    final, meta, idx = build_walk_dataset()
    print(f"walk_dataset shape: {final.shape}")
    print(f"selected subcarrier indices (into the original 128): {idx.tolist()}")
    print(meta.groupby(["date", "person_id"]).size())

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(ARTIFACT_PATH, walk_dataset=final, subcarrier_idx=idx,
              session_dir=meta["session_dir"].to_numpy(), person_id=meta["person_id"].to_numpy(),
              date=meta["date"].to_numpy(), window_start_time_us=meta["window_start_time_us"].to_numpy())
    print(f"saved to {ARTIFACT_PATH}")
