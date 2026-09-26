"""Diagnostic: is Day 2's cross-day failure because Day 2 genuinely carries LESS anjali-vs-barath
signal (128 vs 186 subcarriers -> less information), or because Day 1 and Day 2 are just different
distributions (a transfer/shift problem, not an information-content problem)?

Tests this directly and per-day, independent of any windowing/model choice: per-subcarrier Cohen's d
between anjali and barath, on each day's own native (unpreprocessed) decoded amplitude -- if Day 2 is
genuinely information-poorer, its effect sizes should be systematically smaller than Day 1's. Also
reports raw signal magnitude/RSSI, to check whether Day 2 is a weaker-SNR capture in absolute terms.
"""
from __future__ import annotations

import numpy as np

from ml.data_pipeline.decode_csi import load_session
from ml.data_pipeline.windowing import load_manifest
from ml.training.run_taskB_same_day import cache_session_native


def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean_diff = a.mean(axis=0) - b.mean(axis=0)
    pooled_std = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2)
    return mean_diff / np.maximum(pooled_std, 1e-6)


def day_summary(day: str) -> None:
    manifest = load_manifest()
    day_auth = manifest[(manifest["label"] == "authorized")
                         & manifest["session_dir"].str.contains(f"/{day}/", regex=False)]

    amp_by_person = {}
    rssi_all = []
    for person in sorted(day_auth["person_id"].unique()):
        sessions = sorted(day_auth.loc[day_auth["person_id"] == person, "session_dir"])
        amps = []
        for sd in sessions:
            cache_path = cache_session_native(sd)
            with np.load(cache_path) as d:
                amps.append(d["amplitude"])
                rssi_all.append(d["rssi"])
        amp_by_person[person] = np.concatenate(amps, axis=0)

    anjali, barath = amp_by_person["anjali"], amp_by_person["barath"]
    d = cohens_d(anjali, barath)
    rssi = np.concatenate(rssi_all)

    print(f"\n=== {day} (n_subcarriers={anjali.shape[1]}) ===")
    print(f"  anjali packets={len(anjali)}, barath packets={len(barath)}")
    print(f"  amplitude: mean={np.concatenate([anjali, barath]).mean():.2f}, "
          f"std={np.concatenate([anjali, barath]).std():.2f}, "
          f"max={np.concatenate([anjali, barath]).max():.2f}")
    print(f"  rssi: mean={rssi.mean():.2f} dBm, std={rssi.std():.2f}")
    print(f"  |Cohen's d| per subcarrier: mean={np.abs(d).mean():.3f}, median={np.median(np.abs(d)):.3f}, "
          f"max={np.abs(d).max():.3f}")
    for thresh, label in [(0.2, "small+"), (0.5, "medium+"), (0.8, "large+")]:
        frac = (np.abs(d) > thresh).mean()
        print(f"  subcarriers with |d|>{thresh} ({label}): {(np.abs(d) > thresh).sum()}/{len(d)} ({100*frac:.1f}%)")


def main() -> None:
    for day in ["2026-09-09", "2026-09-10"]:
        day_summary(day)


if __name__ == "__main__":
    main()
