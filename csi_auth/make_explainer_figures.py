"""One-off script generating the illustrative figures for PIPELINE_EXPLAINER.md -- not part of the
analysis pipeline itself, just visuals for the writeup. Uses a real Anjali session.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cleaning import hampel_filter_amplitude
from decode import decode_dominant_bucket, load_npz
from subcarrier_mask import NULL_SUBCARRIERS

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SESSION = r"C:\Users\av9146\Desktop\wifi-sense\wifi-csi-presence-collector\data\authorized\2026-09-24\20260924_164118_anjali\ac276ea55bc8_samples.npz"
BLUE, RED, GRAY = "#2a78d6", "#e34948", "#898781"


def main():
    npz = load_npz(Path(SESSION))
    bucket = decode_dominant_bucket(npz)
    amp = bucket["amplitude"]
    cleaned, frac_flagged = hampel_filter_amplitude(amp)

    # --- Figure 1: raw vs. spike-cleaned amplitude, one subcarrier, first 300 packets ---
    sc = 45  # a calm subcarrier -- avoids the ~64-96 band's separate hardware switching artifact
             # (see FINDINGS.md), which would otherwise be confused with what a "spike" is here
    n = 300
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(amp[:n, sc], color=GRAY, linewidth=1.0, alpha=0.8, label="raw")
    ax.plot(cleaned[:n, sc], color=BLUE, linewidth=1.3, label="Hampel-filtered")
    spikes = np.abs(amp[:n, sc] - cleaned[:n, sc]) > 1e-6
    ax.scatter(np.flatnonzero(spikes), amp[:n, sc][spikes], color=RED, s=18, zorder=5,
               label="flagged as spikes")
    ax.set_xlabel("packet #")
    ax.set_ylabel(f"amplitude, subcarrier {sc}")
    ax.set_title(f"Spike removal (Hampel filter): {100*frac_flagged:.1f}% of this session's samples flagged")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "preprocessing_spike_removal.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: mean amplitude per subcarrier, with the 19 masked (null) bins highlighted ---
    mean_amp = cleaned.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    kept = np.ones(128, dtype=bool)
    kept[NULL_SUBCARRIERS] = False
    ax.plot(np.flatnonzero(kept), mean_amp[kept], color=BLUE, linewidth=1.3, label="kept (109 bins)")
    ax.scatter(NULL_SUBCARRIERS, mean_amp[NULL_SUBCARRIERS], color=RED, s=28, zorder=5,
               label="masked out (19 null guard/DC/pilot bins)")
    ax.set_xlabel("subcarrier index (0-127)")
    ax.set_ylabel("mean amplitude (whole session)")
    ax.set_title("Subcarrier masking: these 19 bins read ~zero in EVERY session, even empty rooms")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "preprocessing_subcarrier_mask.png", dpi=150)
    plt.close(fig)

    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
