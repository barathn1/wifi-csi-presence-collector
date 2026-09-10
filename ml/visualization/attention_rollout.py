"""What the cross-attention transformer (models/transformer_crossattn.py) actually attends to: a
time-by-time heatmap (query packet vs key packet, averaged over heads and over a batch of windows) for
both attention directions (amplitude->phase, phase->amplitude). This is a different axis than
effect_size_heatmap.py / feature_importance.py (which show which SUBCARRIERS matter) -- this shows
which MOMENTS within the 1s window the model leans on, per class.

Trains a small crossattn model itself (fast enough on taskB's ~4800 windows) rather than requiring a
saved checkpoint from run_stage2.py, which doesn't currently persist weights.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import session_disjoint_kfold
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.models.transformer_crossattn import CrossAttentionTransformer
from ml.training.train import train_classifier
from ml.visualization.palette import STATIC_FIG_DIR, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SEQUENTIAL_BLUE

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("blue_seq", SEQUENTIAL_BLUE)


@torch.no_grad()
def average_attention(model: CrossAttentionTransformer, ds, direction: str, max_windows: int = 256) -> np.ndarray:
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    sums, n = None, 0
    for amp, phase, _ in loader:
        weights = model.attention_weights(amp, phase)[direction]  # (B, n_heads, T, T)
        batch_mean = weights.mean(dim=(0, 1)).numpy()  # average over batch and heads -> (T, T)
        sums = batch_mean if sums is None else sums + batch_mean
        n += 1
        if n * amp.shape[0] >= max_windows:
            break
    return sums / n


def plot_attention(matrix: np.ndarray, title: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#fcfcfb")
    im = ax.imshow(matrix, aspect="auto", cmap=SEQUENTIAL_CMAP)
    ax.set_xlabel("key packet index (within 1s window)", color=INK_SECONDARY)
    ax.set_ylabel("query packet index", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.tick_params(colors=INK_MUTED)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("mean attention weight", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)

    STATIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(STATIC_FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"wrote {STATIC_FIG_DIR / out_name}")


def main(epochs: int = 4) -> None:
    index_path = REPO_ROOT / "ml/data_pipeline/cache/window_index_w200_s100.csv"
    window_index = pd.read_csv(index_path)

    full_ds = CsiWindowDataset(window_index, "taskB_identity")
    train_idx, test_idx = next(session_disjoint_kfold(full_ds.index, n_splits=5))
    train_ds = full_ds.subset_by_index_rows(train_idx)
    test_ds = full_ds.subset_by_index_rows(test_idx)

    model = CrossAttentionTransformer(n_subcarriers=186, n_classes=len(full_ds.classes))
    result = train_classifier(model, train_ds, test_ds, epochs=epochs)
    print(f"trained crossattn for attention visualization: test accuracy={result['accuracy']:.4f}")

    for identity in full_ds.classes:
        idx = np.flatnonzero((test_ds.index["person_id"] == identity).values)
        if len(idx) == 0:
            continue
        subset = test_ds.subset_by_index_rows(idx)
        for direction, label in [("amp_attends_phase", "amplitude attends to phase"),
                                  ("phase_attends_amp", "phase attends to amplitude")]:
            matrix = average_attention(model, subset, direction)
            plot_attention(matrix, f"{label}: {identity} (crossattn transformer, taskB)",
                            f"attention_{direction}_{identity}.png")


if __name__ == "__main__":
    main()
