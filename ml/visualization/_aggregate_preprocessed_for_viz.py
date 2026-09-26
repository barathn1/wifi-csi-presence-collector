"""Downsamples _empty_room_preprocessed.json (16MB, one full subcarrier vector per
overlapping 1s-stride clip) into a small payload fit for an HTML chart: a bucketed
subcarrier heatmap, a per-clip scalar mean-amplitude line, and a kept/dropped flag per
clip. Scratch tool -- safe to delete alongside its siblings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TARGET_BUCKETS = 260


def aggregate_day(day: dict) -> dict:
    clips = day["clips"]
    n_clips = len(clips)
    n_sub = day["n_subcarriers"]

    kept_flags = [1 if c["kept"] else 0 for c in clips]
    mean_amp = [round(float(np.mean(c["amp"])), 2) if c["kept"] else -1.0 for c in clips]

    bucket_size = max(1, int(np.ceil(n_clips / TARGET_BUCKETS)))
    n_buckets = int(np.ceil(n_clips / bucket_size))
    heat = np.full((n_buckets, n_sub), -1.0, dtype=np.float64)
    for b in range(n_buckets):
        lo, hi = b * bucket_size, min(n_clips, (b + 1) * bucket_size)
        vecs = [clips[i]["amp"] for i in range(lo, hi) if clips[i]["kept"]]
        if vecs:
            heat[b] = np.mean(np.array(vecs, dtype=np.float64), axis=0)

    # exact clip-index boundaries per session -> bucket-index boundaries
    cum = 0
    boundaries_bucket = []
    for s in day["per_session"]:
        boundaries_bucket.append(cum / bucket_size)
        cum += s["n_clips"]

    return {
        "day": day["day"],
        "n_subcarriers": n_sub,
        "n_sessions": day["n_sessions"],
        "clip_len_s": day["clip_len_s"],
        "stride_s": day["stride_s"],
        "coverage_threshold": day["coverage_threshold"],
        "n_clips_total": day["n_clips_total"],
        "n_clips_kept": day["n_clips_kept"],
        "total_raw_packets": day["total_raw_packets"],
        "total_clean_packets": day["total_clean_packets"],
        "bucket_size_clips": bucket_size,
        "n_buckets": n_buckets,
        "session_boundaries_bucket": boundaries_bucket,
        "heat": heat.round(2).tolist(),
        "kept_flags": kept_flags,
        "mean_amp": mean_amp,
        "per_session": day["per_session"],
    }


def main() -> None:
    raw = json.loads((HERE / "_empty_room_preprocessed.json").read_text())
    out = [aggregate_day(d) for d in raw]
    out_path = HERE / "_empty_room_viz.json"
    out_path.write_text(json.dumps(out))
    print("wrote", out_path, out_path.stat().st_size / 1024, "KB")


if __name__ == "__main__":
    main()
