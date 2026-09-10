"""2D projection (UMAP by default, t-SNE optional) of window-level feature vectors -- later, once the
transformer models exist, the same function takes their learned embeddings instead. Colored by class or
person: shows cluster separation at a glance, without ever plotting a raw CSI trace.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.visualization.palette import STATIC_FIG_DIR, CATEGORICAL, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY

MAX_POINTS = 4000  # keep the projection fast; random subsample above this


def project_2d(X: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    X_scaled = StandardScaler().fit_transform(X)
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed, init="pca")
    return reducer.fit_transform(X_scaled)


def plot_embedding(X: np.ndarray, labels: np.ndarray, title: str, out_name: str, method: str = "umap",
                    seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    if len(X) > MAX_POINTS:
        idx = rng.choice(len(X), MAX_POINTS, replace=False)
        X, labels = X[idx], labels[idx]

    coords = project_2d(X, method=method, seed=seed)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for group in sorted(set(labels)):
        mask = labels == group
        color = CATEGORICAL.get(group, INK_SECONDARY)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6, color=color, label=f"{group} (n={mask.sum()})")

    ax.set_title(title, color=INK_PRIMARY, fontsize=13, loc="left")
    ax.set_xlabel(f"{method} 1", color=INK_SECONDARY)
    ax.set_ylabel(f"{method} 2", color=INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, markerscale=2)

    STATIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(STATIC_FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"wrote {STATIC_FIG_DIR / out_name}")


def main() -> None:
    cache_dir = REPO_ROOT / "ml/data_pipeline/cache"
    window_index = pd.read_csv(cache_dir / "window_index_w200_s100.csv")
    X = np.load(cache_dir / "features_raw_window_index_w200_s100.npy")

    plot_embedding(X, window_index["label"].values,
                    "Window features (raw), colored by class", "embedding_by_class.png")

    auth_mask = (window_index["label"] == "authorized").values
    plot_embedding(X[auth_mask], window_index.loc[auth_mask, "person_id"].values,
                    "Window features (raw), authorized only: anjali vs barath", "embedding_identity.png")


if __name__ == "__main__":
    main()
