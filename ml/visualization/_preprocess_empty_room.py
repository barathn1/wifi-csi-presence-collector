"""Scratch preprocessing for empty-room ("none") sessions, per the exact recipe:

1. Per packet: amplitude = sqrt(real^2 + imag^2) per subcarrier.
2. Drop packets that are incomplete (bad csi_len/offset), corrupted (first_word_invalid
   hardware flag -- see RESEARCH_NOTES.md), or a duplicate device_time_us (keep first
   occurrence, drop the rest).
3. Cut into 3s clips, 1s stride (overlapping).
4. Drop any clip with < 50% of the session's expected packet count in that 3s span
   (expected = metadata avg_rate_hz * 3).

Amplitude needs a fixed subcarrier count to stack into a clip, so -- consistent with
ml/data_pipeline/windowing.py's existing convention -- clip amplitude vectors are built
from each session's dominant csi_len bucket only (computed AFTER the drop step above).
The 50% coverage check itself counts ALL surviving clean packets in the window
(any bucket), since that's what "missing data" means; only the amplitude payload is
bucket-restricted. Not part of the pipeline -- safe to delete.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.data_pipeline.decode_csi import load_session

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "none"
CLIP_LEN_S = 3.0
STRIDE_S = 1.0
COVERAGE_THRESHOLD = 0.5


def clean_packet_mask(npz: dict) -> np.ndarray:
    csi_len = npz["csi_len"].astype(np.int64)
    csi_offset = npz["csi_offset"].astype(np.int64)
    flat_size = npz["csi_flat"].size
    incomplete = (csi_len <= 0) | (csi_len % 2 != 0) | (csi_offset + csi_len > flat_size)
    corrupted = npz["first_word_invalid"].astype(bool)
    return ~(incomplete | corrupted)


def drop_duplicate_timestamps(idx: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times, kind="stable")
    idx_sorted, times_sorted = idx[order], times[order]
    is_first = np.ones(len(times_sorted), dtype=bool)
    is_first[1:] = times_sorted[1:] != times_sorted[:-1]
    return idx_sorted[is_first], times_sorted[is_first]


def decode_amplitude(npz: dict, idx: np.ndarray, csi_len_val: int) -> np.ndarray:
    csi_offset = npz["csi_offset"].astype(np.int64)
    csi_flat = npz["csi_flat"]
    n_sub = csi_len_val // 2
    offs = csi_offset[idx]
    gather = offs[:, None] + np.arange(csi_len_val)[None, :]
    raw = csi_flat[gather].astype(np.float32)
    imag, real = raw[:, 0::2], raw[:, 1::2]
    return np.hypot(real, imag).astype(np.float32)


def process_session(session_dir: Path) -> dict | None:
    session = load_session(session_dir)
    npz, meta = session.npz, session.metadata

    clean = clean_packet_mask(npz)
    clean_idx = np.flatnonzero(clean)
    if len(clean_idx) == 0:
        return None
    n_raw = len(npz["csi_len"])
    n_before_dedup = len(clean_idx)

    clean_idx, clean_times = drop_duplicate_timestamps(clean_idx, npz["device_time_us"][clean_idx].astype(np.int64))
    n_after_dedup = len(clean_idx)

    t0 = clean_times.min()
    t_all_s = (clean_times - t0) / 1e6
    duration_s = float(t_all_s.max())

    csi_len_clean = npz["csi_len"][clean_idx].astype(np.int64)
    lens, counts = np.unique(csi_len_clean, return_counts=True)
    dom_len = int(lens[np.argmax(counts)])
    dom_sel = csi_len_clean == dom_len
    dom_idx = clean_idx[dom_sel]
    dom_t_s = t_all_s[dom_sel]
    n_sub = dom_len // 2
    dom_amp = decode_amplitude(npz, dom_idx, dom_len)

    expected_rate = float(meta.get("avg_rate_hz") or (n_raw / meta["duration_s"]))
    expected_per_clip = expected_rate * CLIP_LEN_S
    min_required = COVERAGE_THRESHOLD * expected_per_clip

    n_clips = max(0, int(np.floor((duration_s - CLIP_LEN_S) / STRIDE_S)) + 1)
    clips = []
    for i in range(n_clips):
        start = i * STRIDE_S
        end = start + CLIP_LEN_S
        lo_all = np.searchsorted(t_all_s, start, side="left")
        hi_all = np.searchsorted(t_all_s, end, side="left")
        pkt_count = int(hi_all - lo_all)
        kept = pkt_count >= min_required

        amp_vec = None
        if kept:
            lo_d = np.searchsorted(dom_t_s, start, side="left")
            hi_d = np.searchsorted(dom_t_s, end, side="left")
            if hi_d > lo_d:
                amp_vec = dom_amp[lo_d:hi_d].mean(axis=0)
        clips.append({
            "start_s": start, "kept": bool(kept and amp_vec is not None),
            "packet_count": pkt_count, "expected": expected_per_clip,
            "amp": amp_vec,
        })

    return {
        "session": session_dir.name,
        "n_raw_packets": n_raw,
        "n_clean_packets": n_after_dedup,
        "n_dropped_bad": n_raw - n_before_dedup,
        "n_dropped_dup": n_before_dedup - n_after_dedup,
        "dominant_len": dom_len,
        "n_subcarriers": n_sub,
        "duration_s": duration_s,
        "expected_rate_hz": expected_rate,
        "clips": clips,
    }


def process_day(day: str) -> dict:
    day_dir = DATA_DIR / day
    session_dirs = sorted(p for p in day_dir.iterdir() if (p / "samples.npz").exists())

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
        "day": day,
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
    days = ["2026-09-09", "2026-09-10"]
    out = [process_day(d) for d in days]
    out_path = Path(__file__).parent / "_empty_room_preprocessed.json"
    out_path.write_text(json.dumps(out))
    for d in out:
        print(f"{d['day']}: {d['n_sessions']} sessions, {d['total_raw_packets']} raw -> "
              f"{d['total_clean_packets']} clean packets, {d['n_clips_kept']}/{d['n_clips_total']} "
              f"clips kept ({100*d['n_clips_kept']/d['n_clips_total']:.1f}%)")
        for s in d["per_session"]:
            print(f"    {s['session']}: raw={s['n_raw']} clean={s['n_clean']} "
                  f"dom_len={s['dominant_len']} clips_kept={s['n_clips_kept']}/{s['n_clips']}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
