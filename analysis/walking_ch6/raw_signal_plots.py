"""Part 1 -- visualize the raw signal.

- CSI amplitude over time, per person, multiple sessions/days overlaid.
- The same subcarrier tracked across different recordings of the same person.
- Anjali vs Barath vs strangers vs empty room, side by side.

Reads directly from the per-session decoded caches built by load_data.build() (via
ml.data_pipeline.windowing.cache_session), not from the windowed feature matrix -- this part is about
what the packet-level signal itself looks like, before any windowing/feature engineering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.walking_ch6.load_data import FIXED_BOARD, restricted_manifest
from analysis.walking_ch6.viz_common import (
    IDENTITY_COLOR, MUTED_TEXT, PRIMARY_TEXT, SECONDARY_TEXT, day_color_map, identity_color, style_axes,
)
from ml.data_pipeline.windowing import cache_session

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PACKETS_SHOWN = 500
EXAMPLE_SUBCARRIER = 64  # mid-band, arbitrary but fixed across all plots for comparability


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


def plot_person_across_days(manifest, person_id: str) -> None:
    """One subplot per day this person has walking sessions, amplitude of EXAMPLE_SUBCARRIER over the
    first N_PACKETS_SHOWN packets of every session that day -- shows within-day AND across-day shape."""
    rows = manifest[manifest["person_id"] == person_id].sort_values(["date", "session_dir"])
    dates = sorted(rows["date"].unique())
    day_color = day_color_map(dates)
    fig, axes = plt.subplots(1, len(dates), figsize=(5 * len(dates), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, date in zip(axes, dates):
        day_rows = rows[rows["date"] == date]
        for _, row in day_rows.iterrows():
            amp = _load_amplitude(row["session_dir"])
            n = min(N_PACKETS_SHOWN, amp.shape[0])
            ax.plot(np.arange(n), amp[:n, EXAMPLE_SUBCARRIER], color=day_color[date], linewidth=0.8,
                     alpha=0.85)
        ax.set_title(date, fontsize=10)
        ax.set_xlabel("packet #")
        style_axes(ax)
    axes[0].set_ylabel(f"amplitude, subcarrier {EXAMPLE_SUBCARRIER}")
    fig.suptitle(f"{person_id}: raw CSI amplitude over time, every walking session by day", color=PRIMARY_TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"raw_timeseries_{person_id}.png", dpi=150)
    plt.close(fig)


def plot_subcarrier_across_recordings(manifest, person_id: str, subcarriers=(20, 64, 100)) -> None:
    """One subplot per subcarrier, all of this person's sessions overlaid (colored by day) -- does the
    SAME subcarrier look similar across different recordings of the same person?"""
    rows = manifest[manifest["person_id"] == person_id].sort_values(["date", "session_dir"])
    day_color = day_color_map(rows["date"].unique())
    fig, axes = plt.subplots(len(subcarriers), 1, figsize=(7, 2.6 * len(subcarriers)), sharex=True)
    axes = np.atleast_1d(axes)
    seen_dates = set()
    for ax, sc in zip(axes, subcarriers):
        for _, row in rows.iterrows():
            amp = _load_amplitude(row["session_dir"])
            n = min(N_PACKETS_SHOWN, amp.shape[0])
            label = row["date"] if row["date"] not in seen_dates else None
            ax.plot(np.arange(n), amp[:n, sc], color=day_color[row["date"]], linewidth=0.8, alpha=0.85,
                     label=label)
            seen_dates.add(row["date"])
        ax.set_ylabel(f"sc {sc}")
        style_axes(ax)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xlabel("packet #")
    fig.suptitle(f"{person_id}: fixed subcarriers across every walking recording", color=PRIMARY_TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"raw_subcarrier_stability_{person_id}.png", dpi=150)
    plt.close(fig)


def plot_people_side_by_side(manifest) -> None:
    """One row per identity group, one column per day available -- same subcarrier, same y-scale,
    first session of that identity/day. Direct visual comparison of Anjali vs Barath vs stranger vs
    empty room in the SAME environment."""
    manifest = manifest.copy()
    manifest["identity"] = manifest.apply(_identity, axis=1)
    groups = ["anjali", "barath", "stranger", "empty_room"]
    dates = sorted(manifest["date"].unique())

    fig, axes = plt.subplots(len(groups), len(dates), figsize=(4.5 * len(dates), 2.4 * len(groups)),
                              sharex=True, sharey=True)
    for gi, group in enumerate(groups):
        if group == "stranger":
            sub = manifest[manifest["identity"].str.startswith("stranger:")]
        else:
            sub = manifest[manifest["identity"] == group]
        for di, date in enumerate(dates):
            ax = axes[gi, di]
            day_sub = sub[sub["date"] == date]
            if len(day_sub) == 0:
                ax.axis("off")
                continue
            row = day_sub.iloc[0]
            amp = _load_amplitude(row["session_dir"])
            n = min(N_PACKETS_SHOWN, amp.shape[0])
            color = IDENTITY_COLOR["stranger"] if group == "stranger" else IDENTITY_COLOR[group]
            ax.plot(np.arange(n), amp[:n, EXAMPLE_SUBCARRIER], color=color, linewidth=0.8)
            label_name = row["person_id"] if group != "empty_room" else "empty room"
            ax.set_title(f"{date}\n{label_name}", fontsize=9, color=SECONDARY_TEXT)
            style_axes(ax)
            if di == 0:
                ax.set_ylabel(group, fontsize=10, color=PRIMARY_TEXT)
    fig.suptitle(f"Anjali vs Barath vs stranger vs empty room -- subcarrier {EXAMPLE_SUBCARRIER}, "
                 f"first {N_PACKETS_SHOWN} packets", color=PRIMARY_TEXT)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "raw_compare_identities.png", dpi=150)
    plt.close(fig)


def plot_mean_spectrum_profile(manifest) -> None:
    """Mean amplitude per subcarrier (whole session average) -- one line per session, colored by
    identity group -- the coarsest possible 'shape' signature, before any time-domain features."""
    manifest = manifest.copy()
    manifest["identity"] = manifest.apply(_identity, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted_labels = set()
    for _, row in manifest.iterrows():
        amp = _load_amplitude(row["session_dir"])
        profile = amp.mean(axis=0)
        group = row["identity"].split(":")[0] if row["identity"].startswith("stranger") else row["identity"]
        color = identity_color(row["identity"])
        label = group if group not in plotted_labels else None
        ax.plot(profile, color=color, linewidth=1.0, alpha=0.75, label=label)
        plotted_labels.add(group)
    ax.set_xlabel("subcarrier index")
    ax.set_ylabel("mean amplitude (whole session)")
    ax.set_title("Mean amplitude-vs-subcarrier profile, every walking session", color=PRIMARY_TEXT)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "raw_mean_spectrum_profile.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    manifest = restricted_manifest()
    plot_person_across_days(manifest, "anjali")
    plot_person_across_days(manifest, "barath")
    plot_subcarrier_across_recordings(manifest, "anjali")
    plot_subcarrier_across_recordings(manifest, "barath")
    plot_people_side_by_side(manifest)
    plot_mean_spectrum_profile(manifest)
    print(f"figures written to {FIG_DIR}")
