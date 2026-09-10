"""RandomForest feature importances, aggregated per subcarrier per channel -- a second, model-agnostic
cross-check against effect_size_heatmap.py's Cohen's d and (later) attention_rollout.py's attention
weights. If all three agree on which subcarriers matter, that's real signal, not a modeling artifact.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.features import feature_slices
from ml.data_pipeline.tasks import TASKS
from ml.models.baselines import make_baseline
from ml.visualization.palette import STATIC_FIG_DIR, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SEQUENTIAL_BLUE

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("blue_seq", SEQUENTIAL_BLUE)


def per_subcarrier_importance(importances: np.ndarray, n_subcarriers: int = 186) -> np.ndarray:
    """(n_features,) importances -> (2, n_subcarriers) matrix, rows=[amp, phase], summed over
    mean/std/skew/kurt stats for that subcarrier (RSSI's 2 scalar features are dropped here)."""
    slices = feature_slices(n_subcarriers)
    out = np.zeros((2, n_subcarriers))
    for row, channel in enumerate(("amp", "phase")):
        for stat in ("mean", "std", "skew", "kurt"):
            out[row] += importances[slices[f"{channel}_{stat}"]]
    return out


def plot_importance_heatmap(matrix: np.ndarray, title: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.2), facecolor="#fcfcfb")
    im = ax.imshow(matrix, aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=0)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["amplitude", "phase"], color=INK_SECONDARY)
    ax.set_xlabel("subcarrier index", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, loc="left")
    ax.tick_params(colors=INK_MUTED)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("summed RF importance", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)

    STATIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(STATIC_FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"wrote {STATIC_FIG_DIR / out_name}")


def main() -> None:
    cache_dir = REPO_ROOT / "ml/data_pipeline/cache"
    window_index = pd.read_csv(cache_dir / "window_index_w200_s100.csv")
    X = np.load(cache_dir / "features_raw_window_index_w200_s100.npy")

    for task_name, task_fn in TASKS.items():
        mask, y = task_fn(window_index)
        model = make_baseline()
        model.fit(X[mask], y)
        matrix = per_subcarrier_importance(model.feature_importances_)
        plot_importance_heatmap(matrix, f"RandomForest feature importance: {task_name} (raw, Day 1)",
                                 f"feature_importance_{task_name}.png")


if __name__ == "__main__":
    main()
