"""Per-subcarrier amplitude/phase 'fingerprint' for each class: mean line + std band across windows,
NOT raw per-packet CSI traces. Answers "what does authorized/unauthorized/none look like on average
and how much does it vary" in one compact figure per channel.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.features import feature_slices
from ml.visualization.palette import STATIC_FIG_DIR, CATEGORICAL, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY



def plot_fingerprint(X: np.ndarray, window_index: pd.DataFrame, channel: str, group_col: str,
                      groups: list[str], title: str, out_name: str) -> None:
    slices = feature_slices()
    mean_cols = X[:, slices[f"{channel}_mean"]]  # (n_windows, n_subcarriers)
    n_sub = mean_cols.shape[1]
    x = np.arange(n_sub)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for group in groups:
        mask = (window_index[group_col] == group).values
        if mask.sum() == 0:
            continue
        vals = mean_cols[mask]  # (n_windows_in_group, n_subcarriers)
        mean_line = vals.mean(axis=0)
        std_band = vals.std(axis=0)
        color = CATEGORICAL.get(group, INK_SECONDARY)
        ax.plot(x, mean_line, label=f"{group} (n={mask.sum()})", color=color, linewidth=2)
        ax.fill_between(x, mean_line - std_band, mean_line + std_band, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("subcarrier index", color=INK_SECONDARY)
    ax.set_ylabel(f"{channel} (mean +/- std across windows)", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, loc="left")
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)

    STATIC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(STATIC_FIG_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"wrote {STATIC_FIG_DIR / out_name}")


def main() -> None:
    cache_dir = REPO_ROOT / "ml/data_pipeline/cache"
    window_index = pd.read_csv(cache_dir / "window_index_w200_s100.csv")
    X_raw = np.load(cache_dir / "features_raw_window_index_w200_s100.npy")

    for channel in ("amp", "phase"):
        plot_fingerprint(
            X_raw, window_index, channel, "label", ["authorized", "unauthorized", "none"],
            title=f"{channel} fingerprint by class (raw, Day 1)",
            out_name=f"fingerprint_{channel}_by_class.png",
        )

    auth_mask = window_index["label"] == "authorized"
    for channel in ("amp", "phase"):
        plot_fingerprint(
            X_raw[auth_mask.values], window_index[auth_mask].reset_index(drop=True), channel,
            "person_id", ["anjali", "barath"],
            title=f"{channel} fingerprint: anjali vs barath (authorized only, raw, Day 1)",
            out_name=f"fingerprint_{channel}_anjali_vs_barath.png",
        )


if __name__ == "__main__":
    main()
