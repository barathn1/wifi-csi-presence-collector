"""EDA / cross-day drift report for the channel-6 person-ID subset (2026-09-15/16/17).

Answers, from THIS data (not from the literature review in RESEARCH_NOTES.md), the checklist raised
for the BiLSTM cross-day-validation work: does CSI packet format change between sessions, is
channel/bandwidth/PHY actually consistent, is the AP/ESP32 link (MAC routing) stable, does amplitude
distribution shift day to day, do packet-rate/timing and person-movement mix differ in ways that could
confound a model, and does this repo's own denoising step (Hampel+Butterworth, arXiv:2507.12854,
already used by `bilstm_ch6_pipeline.py`) actually shrink the day-to-day amplitude drift.

Deliberately reads sessions directly and does NOT write into `data/` or any shared pipeline cache
directory (ablation_ch6/bilstm_ch6/sessions/etc.) -- this can run safely at the same time as a
training job that IS using those caches, with no collision and no extra disk pressure.

    python3 -m ml.evaluation.cross_day_eda_ch6
    python3 -m ml.evaluation.cross_day_eda_ch6 --with-denoised   # also load the ablation cache's
                                                                   # already-denoised amplitude (must
                                                                   # exist on disk already) for a
                                                                   # before/after-denoising comparison
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_pipeline.ablation_ch6_pipeline import TARGET_DATES, build_ch6_manifest_all_labels
from ml.data_pipeline.bilstm_ch6_pipeline import mac_str_to_bytes, select_dominant_packets
from ml.data_pipeline.decode_csi import (
    REPO_ROOT,
    channel_2ghz_freq_mhz,
    csi_len_distribution,
    decode_session_by_bucket,
    load_session,
)
from ml.data_pipeline.time_resample import session_native_rate_hz

TARGET_CHANNEL = 6
REPORT_PATH = REPO_ROOT / "ml/reports/cross_day_ch6_eda_findings.md"


def session_full_summary(session_dir_rel: str, label: str, date: str) -> dict:
    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    npz = session.npz
    meta = session.metadata
    board_mac = meta["board_mac"]

    n_total = len(npz["csi_len"])
    csi_len_dist = csi_len_distribution(npz)

    # MAC pair census over ALL packets (before any filtering) -- how much traffic is even addressed
    # to this ESP32 vs overheard promiscuously from something else.
    src = npz["src_mac"]
    dst = npz["dst_mac"]
    pairs = Counter(
        (":".join(f"{b:02x}" for b in s), ":".join(f"{b:02x}" for b in d))
        for s, d in zip(src.tolist(), dst.tolist())
    )
    board_mac_bytes = mac_str_to_bytes(board_mac)
    frac_to_board = float(np.all(dst == board_mac_bytes, axis=1).mean())

    dominant_idx = select_dominant_packets(npz, board_mac)
    frac_dominant = len(dominant_idx) / n_total

    sub_npz = {k: v[dominant_idx] for k, v in npz.items() if k != "csi_flat"}
    sub_npz["csi_flat"] = npz["csi_flat"]
    buckets = decode_session_by_bucket(sub_npz)
    assert len(buckets) == 1
    bucket = next(iter(buckets.values()))
    amp = bucket["amplitude"].astype(np.float64)  # (n, n_sub)

    device_time_us = sub_npz["device_time_us"].astype(np.int64)
    native_rate_hz = session_native_rate_hz(device_time_us)

    ch_primary = int(npz["channel_primary"][dominant_idx[0]])
    cwb = int(npz["cwb"][dominant_idx[0]])
    mcs = int(npz["mcs"][dominant_idx[0]])
    stbc = int(npz["stbc"][dominant_idx[0]])
    rssi_mean = float(sub_npz["rssi"].astype(np.float64).mean())

    return {
        "session_dir": session_dir_rel, "date": date, "label": label,
        "person_id": meta.get("person_id", ""), "motion": meta.get("motion", ""),
        "notes": meta.get("notes", ""), "ap_source": meta.get("ap_source", ""),
        "board_mac": board_mac,
        "n_packets_total": n_total, "n_packets_dominant": len(dominant_idx),
        "frac_dominant": frac_dominant, "frac_to_board_mac": frac_to_board,
        "n_distinct_mac_pairs": len(pairs), "top_mac_pairs": pairs.most_common(4),
        "n_distinct_csi_len": len(csi_len_dist), "csi_len_dist": csi_len_dist,
        "channel_primary": ch_primary, "channel_freq_mhz": channel_2ghz_freq_mhz(ch_primary),
        "cwb": cwb, "mcs": mcs, "stbc": stbc, "rssi_mean": rssi_mean,
        "native_rate_hz": native_rate_hz, "n_subcarriers": bucket["n_subcarriers"],
        "amp_sum": amp.sum(0), "amp_sumsq": (amp ** 2).sum(0), "n_amp": amp.shape[0],
        "amp_mean_overall": float(amp.mean()), "amp_std_overall": float(amp.std()),
    }


def _agg_amp(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    n = sum(r["n_amp"] for r in rows)
    s = sum(r["amp_sum"] for r in rows)
    sq = sum(r["amp_sumsq"] for r in rows)
    mean = s / n
    std = np.sqrt(np.maximum(sq / n - mean ** 2, 1e-9))
    return mean, std


def _mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def load_denoised_amp(session_dir_rel: str, experiment_name: str) -> np.ndarray | None:
    """Read amplitude straight from an already-populated ablation_ch6 cache slot (never triggers a
    fresh compute) -- read-only, for the before/after-denoising comparison once that background run
    has produced it."""
    safe_name = session_dir_rel.replace("/", "__")
    path = REPO_ROOT / "ml/data_pipeline/cache/ablation_ch6/denoised" / experiment_name / f"{safe_name}.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        return d["amplitude"].astype(np.float64)


def main(with_denoised: bool) -> None:
    manifest = build_ch6_manifest_all_labels()
    print(f"channel-{TARGET_CHANNEL} sessions across {TARGET_DATES}: {len(manifest)}")

    rows = [session_full_summary(r["session_dir"], r["label"], r["date"]) for _, r in manifest.iterrows()]
    df = pd.DataFrame(rows)

    lines = []
    lines.append("# Channel-6 (2026-09-15/16/17) cross-day EDA findings\n")
    lines.append(f"Generated from {len(df)} channel-6 sessions across {TARGET_DATES} "
                  f"(`ml/evaluation/cross_day_eda_ch6.py`). Every number below is measured directly "
                  f"from this repo's own `data/` for the exact subset the BiLSTM cross-day track uses "
                  f"-- nothing here is assumed from metadata.json or prior docs.\n")

    # --- 1. session counts ---
    lines.append("## 1. Session counts by date/label/person\n")
    lines.append("```\n" + df.groupby(["date", "label", "person_id"]).size().to_string() + "\n```\n")

    # --- 2. channel / bandwidth / PHY ---
    lines.append("## 2. Channel / bandwidth / PHY configuration\n")
    phy = df.groupby("date")[["channel_primary", "channel_freq_mhz", "cwb", "mcs", "stbc", "n_subcarriers"]].agg(
        lambda s: sorted(s.unique().tolist()))
    lines.append("Unique values observed per date (all channel-6-filtered sessions already share "
                  "channel_primary=6 by construction -- this checks everything else):\n")
    lines.append("```\n" + phy.to_string() + "\n```\n")
    frac_dom = df.groupby("date")["frac_dominant"].agg(["mean", "min", "max"])
    lines.append("\nFraction of each session's packets kept after MAC+PHY-combo filtering "
                  "(low values would mean the \"one consistent CSI packet configuration\" assumption "
                  "is being violated mid-session):\n")
    lines.append("```\n" + frac_dom.to_string(float_format=lambda v: f"{v:.3f}") + "\n```\n")
    multi_len = df[df["n_distinct_csi_len"] > 1]
    lines.append(f"\nSessions with more than one distinct `csi_len` (raw packet format changing "
                  f"mid-session, before MAC/PHY filtering): {len(multi_len)} / {len(df)}.\n")
    if len(multi_len):
        lines.append("```\n" + multi_len[["session_dir", "csi_len_dist"]].to_string(index=False) + "\n```\n")

    # --- 3. MAC address routing ---
    lines.append("## 3. MAC address routing\n")
    mac_stats = df.groupby("date")[["frac_to_board_mac", "n_distinct_mac_pairs"]].agg(["mean", "min", "max"])
    lines.append("`frac_to_board_mac`: fraction of a session's RAW packets (before filtering) "
                  "addressed to that session's own ESP32 (`dst_mac == board_mac`).\n")
    lines.append("```\n" + mac_stats.to_string(float_format=lambda v: f"{v:.3f}") + "\n```\n")
    n_boards = df["board_mac"].nunique()
    lines.append(f"\nDistinct `board_mac` values across all {len(df)} sessions: {n_boards} "
                  f"({sorted(df['board_mac'].unique())}) -- "
                  f"{'same physical ESP32 used throughout' if n_boards == 1 else 'MULTIPLE ESP32 boards used -- check this is intentional'}.\n")

    # --- 4. packet rate / timing ---
    lines.append("## 4. Packet-rate / timing differences\n")
    rate_by_date = df.groupby("date")["native_rate_hz"].agg(["mean", "std", "min", "max"])
    lines.append("Native packet rate (Hz), post MAC/PHY filtering, per date:\n")
    lines.append("```\n" + rate_by_date.to_string(float_format=lambda v: f"{v:.1f}") + "\n```\n")
    rate_by_date_label = df.groupby(["date", "label"])["native_rate_hz"].agg(["mean", "min", "max", "count"])
    lines.append("\nSame, broken down by label -- this is the exact shape of the previously-found "
                  "packet-rate/window-duration confound (rate correlating with label within a date). "
                  "The BiLSTM pipeline's `time_resample.py` step already resamples every session onto "
                  "one common rate before windowing specifically to neutralize this, but the raw "
                  "numbers below are worth checking directly:\n")
    lines.append("```\n" + rate_by_date_label.to_string(float_format=lambda v: f"{v:.1f}") + "\n```\n")

    # --- 5. movement mix ---
    lines.append("## 5. Person-movement (motion) mix\n")
    motion = df.groupby(["date", "label", "motion"]).size()
    lines.append("```\n" + motion.to_string() + "\n```\n")

    # --- 6. environment / AP placement notes ---
    lines.append("## 6. Environment / AP placement notes (from metadata.json)\n")
    notes = df[df["notes"].astype(str).str.strip() != ""][["session_dir", "date", "notes"]]
    ap_source = df.groupby("date")["ap_source"].agg(lambda s: sorted(s.unique().tolist()))
    lines.append("`ap_source` values per date:\n```\n" + ap_source.to_string() + "\n```\n")
    if len(notes):
        lines.append("\nNon-empty `notes` fields (only source of AP/room-placement info actually "
                      "recorded -- there is no structured placement/distance field in metadata.json):\n")
        lines.append("```\n" + notes.to_string(index=False) + "\n```\n")
    else:
        lines.append("\nNo session across these three dates has a non-empty `notes` field -- physical "
                      "AP/ESP32 placement and any environmental changes between days are **not "
                      "recorded anywhere in this dataset** and cannot be checked from data alone. "
                      "This is a real, unfixable-after-the-fact gap: if placement moved between days, "
                      "there is no metadata trail to detect or control for it, only the amplitude-drift "
                      "numbers below as an indirect symptom.\n")
    rssi = df.groupby("date")["rssi_mean"].agg(["mean", "std", "min", "max"])
    lines.append("\nMean RSSI per date (a cheap proxy for gross link/placement/environment change -- "
                  "a stable link across days should show similar RSSI; a shift suggests something "
                  "physical changed, whether AP position, obstruction, or room occupancy):\n")
    lines.append("```\n" + rssi.to_string(float_format=lambda v: f"{v:.1f}") + "\n```\n")

    # --- 7. amplitude-distribution drift (raw) ---
    lines.append("## 7. Amplitude-distribution drift across days (raw, pre-denoising)\n")
    lines.append("Per-subcarrier amplitude mean/std, pooled over every dominant packet of every "
                  "channel-6 session for that date+label (NOT windowed -- uses all data for a tight "
                  "estimate), compared pairwise across the three dates.\n")
    for label in ["none", "authorized", "unauthorized"]:
        sub = df[df["label"] == label]
        by_date = {d: sub[sub["date"] == d].to_dict("records") for d in TARGET_DATES}
        baselines = {d: _agg_amp(rows) for d, rows in by_date.items() if rows}
        lines.append(f"\n**{label}** (n sessions per date: "
                      f"{ {d: len(r) for d, r in by_date.items()} }):\n")
        dates_present = list(baselines.keys())
        if len(dates_present) < 2:
            lines.append("  only one date has sessions for this label -- no pairwise comparison possible.\n")
            continue
        table = []
        for i in range(len(dates_present)):
            for j in range(i + 1, len(dates_present)):
                d1, d2 = dates_present[i], dates_present[j]
                m1, s1 = baselines[d1]
                m2, s2 = baselines[d2]
                table.append({
                    "pair": f"{d1} vs {d2}",
                    "amp_mean |diff| (mean over subcarriers)": _mean_abs(m1, m2),
                    "amp_mean |diff| (max over subcarriers)": float(np.abs(m1 - m2).max()),
                    "amp_std |diff| (mean over subcarriers)": _mean_abs(s1, s2),
                    "rel_mean_shift_%": 100 * _mean_abs(m1, m2) / max(float((m1 + m2).mean() / 2), 1e-6),
                })
        lines.append("```\n" + pd.DataFrame(table).to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n```\n")

    if with_denoised:
        lines.append("## 8. Amplitude-distribution drift: raw vs after this repo's denoising "
                      "(Hampel + Butterworth, arXiv:2507.12854 recipe)\n")
        lines.append("Reads already-computed cache from the `E1_raw` (no denoising) and "
                      "`E9_paper_hampel_butterworth` (Hampel window=15/n_sigmas=3 + 5th-order "
                      "Butterworth low-pass cutoff=10Hz) ablation experiments -- read-only, does not "
                      "trigger any new computation, so this section is skipped/partial if that run "
                      "hasn't produced a given session's cache yet.\n")
        for exp_name, exp_label in [("E1_raw", "raw"), ("E9_paper_hampel_butterworth", "denoised")]:
            lines.append(f"\n### {exp_label} ({exp_name})\n")
            for label in ["none", "authorized", "unauthorized"]:
                sub = df[df["label"] == label]
                by_date_amp = {}
                for d in TARGET_DATES:
                    sessions = sub[sub["date"] == d]["session_dir"].tolist()
                    amps = [load_denoised_amp(s, exp_name) for s in sessions]
                    amps = [a for a in amps if a is not None]
                    if amps:
                        by_date_amp[d] = amps
                if len(by_date_amp) < 2:
                    lines.append(f"  {label}: insufficient cached data for a pairwise comparison "
                                  f"({ {d: len(v) for d, v in by_date_amp.items()} }).\n")
                    continue
                baselines = {}
                for d, amps in by_date_amp.items():
                    n = sum(a.shape[0] for a in amps)
                    s = sum(a.sum(0) for a in amps)
                    sq = sum((a ** 2).sum(0) for a in amps)
                    mean = s / n
                    std = np.sqrt(np.maximum(sq / n - mean ** 2, 1e-9))
                    baselines[d] = (mean, std)
                dates_present = list(baselines.keys())
                table = []
                for i in range(len(dates_present)):
                    for j in range(i + 1, len(dates_present)):
                        d1, d2 = dates_present[i], dates_present[j]
                        m1, _ = baselines[d1]
                        m2, _ = baselines[d2]
                        table.append({"pair": f"{d1} vs {d2}", "amp_mean |diff| (mean)": _mean_abs(m1, m2)})
                lines.append(f"  **{label}**:\n```\n" +
                              pd.DataFrame(table).to_string(index=False, float_format=lambda v: f"{v:.4f}") +
                              "\n```\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--with-denoised", action="store_true",
                   help="also compare against already-cached E1_raw/E9_paper_hampel_butterworth "
                        "denoised amplitude (read-only, run the ablation study first)")
    args = p.parse_args()
    main(with_denoised=args.with_denoised)
