"""Separates two different signals that were conflated in the pooled Cohen's d check:
(A) "person-distinguishing info" -- anjali vs barath, WITHIN one motion condition (isolates identity
    signal from motion, since pooling standing+walking already proved to wreck accuracy).
(B) "gait info" -- walking vs standing, pooled across people -- how much the signal changes from
    motion itself, regardless of who's doing it.
Computed per day, on native (unpreprocessed) decoded amplitude, to answer: which day carries more of
each, independent of any model/preprocessing choice.
"""
from __future__ import annotations

import numpy as np

from ml.data_pipeline.windowing import load_manifest
from ml.training.run_taskB_same_day import cache_session_native


def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean_diff = a.mean(axis=0) - b.mean(axis=0)
    pooled_std = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2)
    return mean_diff / np.maximum(pooled_std, 1e-6)


def report(label: str, d: np.ndarray) -> None:
    ad = np.abs(d)
    n = len(d)
    print(f"    {label}: mean|d|={ad.mean():.3f} median={np.median(ad):.3f} max={ad.max():.3f} "
          f"medium+({'>'}0.5)={np.sum(ad > 0.5)}/{n} large+(>0.8)={np.sum(ad > 0.8)}/{n}")


def load_amp(sessions: list[str]) -> np.ndarray:
    parts = []
    for sd in sessions:
        cache_path = cache_session_native(sd)
        with np.load(cache_path) as d:
            parts.append(d["amplitude"])
    return np.concatenate(parts, axis=0)


def day_report(day: str) -> None:
    manifest = load_manifest()
    day_auth = manifest[(manifest["label"] == "authorized")
                         & manifest["session_dir"].str.contains(f"/{day}/", regex=False)]

    amp = {}
    for person in ("anjali", "barath"):
        for motion in ("standing", "walking"):
            sessions = sorted(day_auth.loc[(day_auth["person_id"] == person) & (day_auth["motion"] == motion),
                                            "session_dir"])
            amp[(person, motion)] = load_amp(sessions) if sessions else np.empty((0, 0))

    n_sub = next(a.shape[1] for a in amp.values() if a.size)
    print(f"\n=== {day} (n_subcarriers={n_sub}) ===")
    for (person, motion), a in amp.items():
        print(f"    {person}/{motion}: {len(a)} packets")

    print("  (A) person-distinguishing info, WITHIN motion (anjali vs barath):")
    if amp[("anjali", "standing")].size and amp[("barath", "standing")].size:
        report("standing", cohens_d(amp[("anjali", "standing")], amp[("barath", "standing")]))
    if amp[("anjali", "walking")].size and amp[("barath", "walking")].size:
        report("walking ", cohens_d(amp[("anjali", "walking")], amp[("barath", "walking")]))

    print("  (B) gait info, WITHIN person (walking vs standing):")
    if amp[("anjali", "walking")].size and amp[("anjali", "standing")].size:
        report("anjali  ", cohens_d(amp[("anjali", "walking")], amp[("anjali", "standing")]))
    if amp[("barath", "walking")].size and amp[("barath", "standing")].size:
        report("barath  ", cohens_d(amp[("barath", "walking")], amp[("barath", "standing")]))

    all_walking = np.concatenate([amp[("anjali", "walking")], amp[("barath", "walking")]])
    all_standing = np.concatenate([amp[("anjali", "standing")], amp[("barath", "standing")]])
    report("pooled  ", cohens_d(all_walking, all_standing))


def main() -> None:
    for day in ["2026-09-09", "2026-09-10"]:
        day_report(day)


if __name__ == "__main__":
    main()
