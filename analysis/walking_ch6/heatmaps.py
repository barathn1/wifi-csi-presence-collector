"""Part 2 -- CSI heatmaps: x=packet index, y=subcarrier, color=amplitude.

- Per-person grids: same person, every session, split out by day.
- One direct comparison grid: Anjali vs Barath vs stranger vs empty room, same days, same color scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.walking_ch6.load_data import FIXED_BOARD, restricted_manifest
from analysis.walking_ch6.viz_common import BLUE_SEQUENTIAL, PRIMARY_TEXT, SECONDARY_TEXT
from ml.data_pipeline.windowing import cache_session

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PACKETS_SHOWN = 1000
VMAX = 130  # shared color scale across every heatmap -- absolute amplitude comparisons are meaningless otherwise


def _load_amplitude(session_dir: str) -> np.ndarray:
    path = cache_session(session_dir, mode="resampled", board_mac=FIXED_BOARD)
    with np.load(path) as d:
        return np.array(d["amplitude"])


def _identity(row) -> str:
    if row["label"] == "none":
        return "empty_room"
    if row["label"] == "authorized":
        return row["person_id"]
    return f"stranger:{row['person_id']}"


def _heatmap(ax, amp: np.ndarray, title: str) -> None:
    n = min(N_PACKETS_SHOWN, amp.shape[0])
    im = ax.imshow(amp[:n].T, aspect="auto", cmap=BLUE_SEQUENTIAL, vmin=0, vmax=VMAX, origin="lower",
                   interpolation="nearest")
    ax.set_title(title, fontsize=9, color=SECONDARY_TEXT)
    return im


def plot_person_grid(manifest, person_id: str) -> None:
    rows = manifest[manifest["person_id"] == person_id].sort_values(["date", "session_dir"])
    n = len(rows)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    im = None
    for i, (_, row) in enumerate(rows.iterrows()):
        ax = axes[i // ncols][i % ncols]
        amp = _load_amplitude(row["session_dir"])
        session_name = row["session_dir"].split("/")[-1]
        im = _heatmap(ax, amp, f"{row['date']}\n{session_name}")
        if i % ncols == 0:
            ax.set_ylabel("subcarrier")
        if i // ncols == nrows - 1:
            ax.set_xlabel("packet #")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{person_id}: CSI amplitude heatmap, every walking session", color=PRIMARY_TEXT)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="amplitude", shrink=0.6)
    fig.savefig(FIG_DIR / f"heatmap_grid_{person_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_identity_comparison_grid(manifest) -> None:
    manifest = manifest.copy()
    manifest["identity"] = manifest.apply(_identity, axis=1)
    groups = ["anjali", "barath", "stranger", "empty_room"]
    dates = sorted(manifest["date"].unique())

    fig, axes = plt.subplots(len(groups), len(dates), figsize=(4.2 * len(dates), 3.0 * len(groups)),
                              squeeze=False)
    im = None
    for gi, group in enumerate(groups):
        sub = (manifest[manifest["identity"].str.startswith("stranger:")] if group == "stranger"
               else manifest[manifest["identity"] == group])
        for di, date in enumerate(dates):
            ax = axes[gi][di]
            day_sub = sub[sub["date"] == date]
            if len(day_sub) == 0:
                ax.axis("off")
                continue
            row = day_sub.iloc[0]
            amp = _load_amplitude(row["session_dir"])
            label_name = row["person_id"] if group != "empty_room" else "empty room"
            im = _heatmap(ax, amp, f"{date} -- {label_name}")
            if di == 0:
                ax.set_ylabel(group)
            if gi == len(groups) - 1:
                ax.set_xlabel("packet #")
    fig.suptitle("Same color scale throughout: Anjali vs Barath vs stranger vs empty room",
                 color=PRIMARY_TEXT)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="amplitude", shrink=0.6)
    fig.savefig(FIG_DIR / "heatmap_identity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_identity_comparison_grid_residual(manifest) -> None:
    """Same layout as plot_identity_comparison_grid, but each panel has its own per-subcarrier mean
    (that session's static gain pattern -- antenna/hardware response) subtracted first. The raw heatmaps
    are dominated by an identical vertical-stripe pattern present even in empty-room recordings (a fixed
    per-subcarrier hardware artifact, not a person signature); subtracting it reveals whatever dynamic
    residual structure is left, which is where a real person signature would have to live."""
    manifest = manifest.copy()
    manifest["identity"] = manifest.apply(_identity, axis=1)
    groups = ["anjali", "barath", "stranger", "empty_room"]
    dates = sorted(manifest["date"].unique())

    fig, axes = plt.subplots(len(groups), len(dates), figsize=(4.2 * len(dates), 3.0 * len(groups)),
                              squeeze=False)
    im = None
    for gi, group in enumerate(groups):
        sub = (manifest[manifest["identity"].str.startswith("stranger:")] if group == "stranger"
               else manifest[manifest["identity"] == group])
        for di, date in enumerate(dates):
            ax = axes[gi][di]
            day_sub = sub[sub["date"] == date]
            if len(day_sub) == 0:
                ax.axis("off")
                continue
            row = day_sub.iloc[0]
            amp = _load_amplitude(row["session_dir"])
            n = min(N_PACKETS_SHOWN, amp.shape[0])
            residual = amp[:n] - amp[:n].mean(axis=0, keepdims=True)
            label_name = row["person_id"] if group != "empty_room" else "empty room"
            im = ax.imshow(residual.T, aspect="auto", cmap="RdBu_r", vmin=-15, vmax=15, origin="lower",
                            interpolation="nearest")
            ax.set_title(f"{date} -- {label_name}", fontsize=9, color=SECONDARY_TEXT)
            if di == 0:
                ax.set_ylabel(group)
            if gi == len(groups) - 1:
                ax.set_xlabel("packet #")
    fig.suptitle("Residual (session mean per subcarrier subtracted) -- what's left after removing the\n"
                 "static hardware gain pattern", color=PRIMARY_TEXT)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="amplitude - session mean", shrink=0.6)
    fig.savefig(FIG_DIR / "heatmap_identity_comparison_residual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    manifest = restricted_manifest()
    plot_person_grid(manifest, "anjali")
    plot_person_grid(manifest, "barath")
    plot_identity_comparison_grid(manifest)
    plot_identity_comparison_grid_residual(manifest)
    print(f"figures written to {FIG_DIR}")
