"""Part 3 -- feature-space separability: do windows cluster by PERSON, or by DAY/SESSION?

PCA and UMAP on the standardized handcrafted feature matrix (mean/std/skew/kurtosis per subcarrier x
amplitude/phase + rssi stats -- ml.data_pipeline.features), colored once by identity and once by day, so
the two hypotheses are directly comparable on the same embedding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analysis.standing_ch6.load_data import WINDOW_INDEX_PATH, FEATURES_PATH, build
from analysis.standing_ch6.viz_common import DAY_COLOR, IDENTITY_COLOR, PRIMARY_TEXT, identity_color, style_axes

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

MAX_POINTS_PER_SESSION = 60  # subsample overlapping windows so scatter plots aren't dominated by density
RNG = np.random.default_rng(0)


def subsample(window_index: pd.DataFrame, X: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    keep = []
    for _, group in window_index.groupby("session_dir"):
        idx = group.index.to_numpy()
        if len(idx) > MAX_POINTS_PER_SESSION:
            idx = RNG.choice(idx, size=MAX_POINTS_PER_SESSION, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    keep.sort()
    return window_index.loc[keep].reset_index(drop=True), X[keep]


def embed(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0).fit_transform(Xs)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=0, n_neighbors=15, min_dist=0.1)
        um = reducer.fit_transform(Xs)
    except ImportError:
        from sklearn.manifold import TSNE
        um = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(Xs)
    return pca, um


def plot_by_identity(window_index, pca, um) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plotted = set()
    for i, (emb, ax, title) in enumerate([(pca, axes[0], "PCA"), (um, axes[1], "UMAP")]):
        for identity in window_index["identity"].unique():
            mask = (window_index["identity"] == identity).values
            group = identity.split(":")[0] if identity.startswith("stranger") else identity
            color = identity_color(identity)
            label = group if (i == 0 and group not in plotted) else None
            ax.scatter(emb[mask, 0], emb[mask, 1], s=10, color=color, alpha=0.55, label=label,
                       edgecolors="none")
            if i == 0:
                plotted.add(group)
        ax.set_title(title, color=PRIMARY_TEXT)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Colored by identity -- do points cluster by PERSON?", color=PRIMARY_TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_space_by_identity.png", dpi=150)
    plt.close(fig)


def plot_by_day(window_index, pca, um) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for i, (emb, ax, title) in enumerate([(pca, axes[0], "PCA"), (um, axes[1], "UMAP")]):
        for date in sorted(window_index["date"].unique()):
            mask = (window_index["date"] == date).values
            ax.scatter(emb[mask, 0], emb[mask, 1], s=10, color=DAY_COLOR[date], alpha=0.55,
                       label=date if i == 0 else None, edgecolors="none")
        ax.set_title(title, color=PRIMARY_TEXT)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Colored by DAY -- do points cluster by session/day instead?", color=PRIMARY_TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_space_by_day.png", dpi=150)
    plt.close(fig)


def plot_by_session_within_person(window_index, pca, um, person_id: str) -> None:
    """Restricts to one person's windows only -- if same-person/different-session points don't overlap,
    session identity (not person identity) is what the embedding is actually capturing."""
    mask_p = (window_index["identity"] == person_id).values
    sessions = sorted(window_index.loc[mask_p, "session_dir"].unique())
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for i, (emb, ax, title) in enumerate([(pca, axes[0], "PCA"), (um, axes[1], "UMAP")]):
        for si, session in enumerate(sessions):
            mask = mask_p & (window_index["session_dir"] == session).values
            date = window_index.loc[mask, "date"].iloc[0]
            ax.scatter(emb[mask, 0], emb[mask, 1], s=12, color=cmap(si % 10), alpha=0.7,
                       label=f"{date} {session.split('/')[-1]}" if i == 0 else None, edgecolors="none")
        ax.set_title(title, color=PRIMARY_TEXT)
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"{person_id} only, colored by SESSION -- how much does one person's own data move around?",
                 color=PRIMARY_TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"feature_space_within_{person_id}_by_session.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    window_index, X = build()
    window_index, X = subsample(window_index, X)
    pca, um = embed(X)
    np.savez(FIG_DIR.parent / "cache" / "embeddings.npz", pca=pca, umap=um)
    window_index.to_csv(FIG_DIR.parent / "cache" / "embedding_window_index.csv", index=False)

    plot_by_identity(window_index, pca, um)
    plot_by_day(window_index, pca, um)
    plot_by_session_within_person(window_index, pca, um, "anjali")
    plot_by_session_within_person(window_index, pca, um, "barath")
    print(f"figures written to {FIG_DIR}")
