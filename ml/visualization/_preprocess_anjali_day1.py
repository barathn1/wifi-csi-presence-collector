"""Same exact preprocessing recipe as _preprocess_empty_room.py, applied to Anjali's
authorized Day 1 (2026-09-09) sessions instead of the empty-room ('none') sessions.
Scratch tool -- safe to delete alongside its siblings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.data_pipeline.decode_csi import load_session
from ml.visualization._preprocess_empty_room import (
    CLIP_LEN_S, STRIDE_S, COVERAGE_THRESHOLD,
    clean_packet_mask, drop_duplicate_timestamps, decode_amplitude, process_session,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DAY_DIR = REPO_ROOT / "data" / "authorized" / "2026-09-09"


def process_person_day(day_dir: Path, person: str) -> dict:
    session_dirs = sorted(p for p in day_dir.iterdir()
                           if p.name.endswith(f"_{person}") and (p / "samples.npz").exists())

    results = [r for r in (process_session(sd) for sd in session_dirs) if r is not None]
    n_sub_ref = max(set(r["n_subcarriers"] for r in results),
                     key=lambda v: sum(1 for r in results if r["n_subcarriers"] == v))
    results = [r for r in results if r["n_subcarriers"] == n_sub_ref]

    cum_offset = 0.0
    boundaries = []
    flat_clips = []
    for r in results:
        boundaries.append(cum_offset)
        for c in r["clips"]:
            flat_clips.append({
                "t": c["start_s"] + cum_offset,
                "kept": c["kept"],
                "packet_count": c["packet_count"],
                "expected": c["expected"],
                "amp": None if c["amp"] is None else c["amp"].round(2).tolist(),
            })
        cum_offset += r["duration_s"] + 1.0

    n_total = len(flat_clips)
    n_kept = sum(1 for c in flat_clips if c["kept"])
    total_raw = sum(r["n_raw_packets"] for r in results)
    total_clean = sum(r["n_clean_packets"] for r in results)

    return {
        "day": "2026-09-09", "person": person,
        "n_subcarriers": n_sub_ref,
        "n_sessions": len(results),
        "session_boundaries_s": boundaries,
        "clip_len_s": CLIP_LEN_S,
        "stride_s": STRIDE_S,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "n_clips_total": n_total,
        "n_clips_kept": n_kept,
        "total_raw_packets": total_raw,
        "total_clean_packets": total_clean,
        "clips": flat_clips,
        "per_session": [{
            "session": r["session"], "n_raw": r["n_raw_packets"], "n_clean": r["n_clean_packets"],
            "dominant_len": r["dominant_len"], "n_clips": len(r["clips"]),
            "n_clips_kept": sum(1 for c in r["clips"] if c["kept"]),
        } for r in results],
    }


def main() -> None:
    out = process_person_day(DAY_DIR, "anjali")
    out_path = Path(__file__).parent / "_anjali_day1_preprocessed.json"
    out_path.write_text(json.dumps([out]))
    print(f"anjali day1: {out['n_sessions']} sessions, {out['total_raw_packets']} raw -> "
          f"{out['total_clean_packets']} clean packets, {out['n_clips_kept']}/{out['n_clips_total']} "
          f"clips kept ({100*out['n_clips_kept']/out['n_clips_total']:.1f}%)")
    for s in out["per_session"]:
        print(f"    {s['session']}: raw={s['n_raw']} clean={s['n_clean']} "
              f"dom_len={s['dominant_len']} clips_kept={s['n_clips_kept']}/{s['n_clips']}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
