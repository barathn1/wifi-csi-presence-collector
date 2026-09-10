"""Cohen's d per subcarrier x class-pair -- the direct answer to "which subcarriers actually
differentiate authorized from unauthorized/none (or anjali from barath)". Rendered for raw AND
Variant-A-calibrated features side by side, to show whether empty-room calibration helps or hurts
separability (the user's calibration hypothesis, visualized rather than just measured by accuracy).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.features import feature_slices
from ml.visualization.palette import STATIC_FIG_DIR, DIVERGING, INK_MUTED, INK_PRIMARY, INK_SECONDARY

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", [DIVERGING["neg"], DIVERGING["mid"], DIVERGING["pos"]]
)


def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a, b: (n_samples, n_subcarriers) -> (n_subcarriers,) effect size, one value per subcarrier."""
    mean_diff = a.mean(axis=0) - b.mean(axis=0)
    pooled_std = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2)
    return mean_diff / np.maximum(pooled_std, 1e-6)


def build_effect_size_matrix(X: np.ndarray, window_index: pd.DataFrame, channel: str,
                              pairs: list[tuple[str, str, str]]) -> tuple[np.ndarray, list[str]]:
    """pairs: list of (group_col, group_a, group_b). Returns (n_pairs, n_subcarriers) matrix + row labels."""
    slices = feature_slices()
    mean_cols = X[:, slices[f"{channel}_mean"]]

    rows, labels = [], []
    for group_col, a, b in pairs:
        mask_a = (window_index[group_col] == a).values
        mask_b = (window_index[group_col] == b).values
        rows.append(cohens_d(mean_cols[mask_a], mean_cols[mask_b]))
        labels.append(f"{a} vs {b}")
    return np.stack(rows), labels


def plot_heatmap(matrix: np.ndarray, row_labels: list[str], title: str, out_name: str) -> None:
    vmax = np.abs(matrix).max()
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(row_labels) + 1.5), facecolor="#fcfcfb")
    im = ax.imshow(matrix, aspect="auto", cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, color=INK_SECONDARY)
    ax.set_xlabel("subcarrier index", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, loc="left")
    ax.tick_params(colors=INK_MUTED)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)

    STATIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(STATIC_FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"wrote {STATIC_FIG_DIR / out_name}")


def main() -> None:
    cache_dir = REPO_ROOT / "ml/data_pipeline/cache"
    window_index = pd.read_csv(cache_dir / "window_index_w200_s100.csv")

    class_pairs = [
        ("label", "authorized", "none"),
        ("label", "authorized", "unauthorized"),
        ("label", "unauthorized", "none"),
    ]
    auth_only_index = window_index[window_index["label"] == "authorized"].reset_index(drop=True)
    identity_pairs = [("person_id", "anjali", "barath")]

    for preprocessing, feat_file in [("raw", "features_raw_window_index_w200_s100.npy"),
                                      ("calibA", "features_calibA_window_index_w200_s100.npy")]:
        feat_path = cache_dir / feat_file
        if not feat_path.exists():
            print(f"skipping {preprocessing}: not built yet")
            continue
        X = np.load(feat_path)
        auth_only_X = X[(window_index["label"] == "authorized").values]

        for channel in ("amp", "phase"):
            matrix, labels = build_effect_size_matrix(X, window_index, channel, class_pairs)
            plot_heatmap(matrix, labels, f"{channel} effect size by class pair ({preprocessing}, Day 1)",
                         f"effect_size_{channel}_class_{preprocessing}.png")

            id_matrix, id_labels = build_effect_size_matrix(auth_only_X, auth_only_index, channel, identity_pairs)
            plot_heatmap(id_matrix, id_labels, f"{channel} effect size: anjali vs barath ({preprocessing})",
                         f"effect_size_{channel}_identity_{preprocessing}.png")


if __name__ == "__main__":
    main()
