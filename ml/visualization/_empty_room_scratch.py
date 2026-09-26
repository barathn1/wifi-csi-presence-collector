"""Scratch analysis: aggregate raw CSI amplitude for every 'none' (empty room) session,
per day, into a time-binned heatmap (subcarrier x time) + mean-amplitude line, for
an ad-hoc HTML visualization. Not part of the pipeline -- safe to delete.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.data_pipeline.decode_csi import load_session, csi_len_distribution

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "none"
BIN_WIDTH_S = 15.0


def dominant_bucket_amplitude(npz: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (device_time_s relative to first sample, amplitude[n_packets, n_sub], n_sub)
    restricted to the session's dominant csi_len bucket."""
    csi_len = npz["csi_len"]
    dist = csi_len_distribution(npz)
    dom_len = max(dist.items(), key=lambda kv: kv[1])[0]
    idx = np.flatnonzero(csi_len == dom_len)

    csi_offset = npz["csi_offset"].astype(np.int64)
    csi_flat = npz["csi_flat"]
    n_sub = dom_len // 2
    offs = csi_offset[idx]
    gather_idx = offs[:, None] + np.arange(dom_len)[None, :]
    raw = csi_flat[gather_idx].astype(np.float32)
    imag = raw[:, 0::2]
    real = raw[:, 1::2]
    amplitude = np.hypot(real, imag).astype(np.float32)

    t_us = npz["device_time_us"][idx].astype(np.int64)
    t_s = (t_us - int(npz["device_time_us"][0])) / 1e6
    return t_s, amplitude, n_sub


def process_day(day: str) -> dict:
    day_dir = DATA_DIR / day
    session_dirs = sorted(p for p in day_dir.iterdir() if (p / "samples.npz").exists())

    n_sub_ref = None
    cum_offset = 0.0
    boundaries = []
    all_bin_idx = []
    all_amp_mean = []  # per-packet mean amplitude across subcarriers (for the line chart)
    all_amp_full = []  # per-packet full subcarrier amplitude (for the heatmap)
    all_t = []

    for sd in session_dirs:
        session = load_session(sd)
        t_s, amplitude, n_sub = dominant_bucket_amplitude(session.npz)
        if n_sub_ref is None:
            n_sub_ref = n_sub
        if n_sub != n_sub_ref:
            continue  # skip sessions whose dominant mode doesn't match the day's mode
        boundaries.append(cum_offset)
        all_t.append(t_s + cum_offset)
        all_amp_mean.append(amplitude.mean(axis=1))
        all_amp_full.append(amplitude)
        cum_offset += t_s.max() + 1.0  # +1s pad so sessions don't visually overlap

    t = np.concatenate(all_t)
    amp_mean = np.concatenate(all_amp_mean)
    amp_full = np.concatenate(all_amp_full, axis=0)

    n_bins = int(np.ceil(t.max() / BIN_WIDTH_S)) + 1
    bin_idx = np.minimum((t // BIN_WIDTH_S).astype(np.int64), n_bins - 1)

    heat = np.zeros((n_bins, n_sub_ref), dtype=np.float64)
    line_mean = np.zeros(n_bins, dtype=np.float64)
    line_std = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_idx == b
        c = int(mask.sum())
        counts[b] = c
        if c == 0:
            heat[b] = np.nan
            line_mean[b] = np.nan
            line_std[b] = np.nan
            continue
        heat[b] = amp_full[mask].mean(axis=0)
        line_mean[b] = amp_mean[mask].mean()
        line_std[b] = amp_mean[mask].std()

    return {
        "day": day,
        "n_subcarriers": n_sub_ref,
        "bin_width_s": BIN_WIDTH_S,
        "n_bins": n_bins,
        "total_packets": int(len(t)),
        "n_sessions": len(all_t),
        "session_boundaries_s": boundaries,
        "heat": np.nan_to_num(heat, nan=-1.0).round(2).tolist(),
        "line_mean": np.nan_to_num(line_mean, nan=-1.0).round(2).tolist(),
        "line_std": np.nan_to_num(line_std, nan=-1.0).round(2).tolist(),
    }


def main() -> None:
    days = ["2026-09-09", "2026-09-10"]
    out = [process_day(d) for d in days]
    out_path = Path(__file__).parent / "_empty_room_scratch.json"
    out_path.write_text(json.dumps(out))
    for d in out:
        print(d["day"], "subcarriers=", d["n_subcarriers"], "bins=", d["n_bins"],
              "sessions=", d["n_sessions"], "packets=", d["total_packets"])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
