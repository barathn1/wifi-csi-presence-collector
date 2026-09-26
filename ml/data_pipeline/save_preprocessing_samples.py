"""Save a small before/after CSI sample (raw -> cleaned -> smoothed -> normalized) for visual
inspection, for one representative channel-6 session, under the two "full chain" experiments
(E7: hampel+smoothing+normalization, E8: moving_iqr+smoothing+normalization) -- the only two
experiments in the ablation that exercise every stage of the requested flow.

Normalization here uses this ONE session's own mean/std (there's no train/test split in a
standalone sample dump) -- for the real ablation's actual reported metrics, normalization stats
come only from each fold's training dates (`ablation_ch6_pipeline.compute_train_normalization_stats`).
This script is for a visual sanity check of what each stage does to the signal, not a metrics run.

Writes, per experiment: a CSV of (packet_idx, subcarrier, raw, cleaned, smoothed, normalized) for a
couple of representative subcarriers, and a 4-row PNG plot of the same.

    python3 -m ml.data_pipeline.save_preprocessing_samples
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.data_pipeline.ablation_ch6_pipeline import build_ch6_manifest_all_labels
from ml.data_pipeline.ablation_preprocessing import EXPERIMENTS, apply_outlier_removal, apply_smoothing
from ml.data_pipeline.bilstm_ch6_pipeline import N_SUB_EXPECTED, select_dominant_packets
from ml.data_pipeline.decode_csi import REPO_ROOT, decode_session_by_bucket, load_session

OUT_DIR = REPO_ROOT / "ml/reports/preprocessing_samples"
SAMPLE_EXPERIMENTS = ["E7_hampel_smooth_norm", "E8_moving_iqr_smooth_norm"]
N_PACKETS_SHOWN = 300
SUBCARRIERS_SHOWN = [0, 64]
ACCENT = "#3B6FA0"


def load_raw_amplitude(session_dir_rel: str) -> np.ndarray:
    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    dominant_idx = select_dominant_packets(session.npz, session.metadata["board_mac"])
    sub_npz = {k: v[dominant_idx] for k, v in session.npz.items() if k != "csi_flat"}
    sub_npz["csi_flat"] = session.npz["csi_flat"]
    buckets = decode_session_by_bucket(sub_npz)
    bucket = next(iter(buckets.values()))
    assert bucket["n_subcarriers"] == N_SUB_EXPECTED
    return bucket["amplitude"].astype(np.float32)


def build_sample(session_dir_rel: str, experiment_name: str) -> pd.DataFrame:
    config = EXPERIMENTS[experiment_name]
    raw = load_raw_amplitude(session_dir_rel)[:N_PACKETS_SHOWN]

    cleaned = apply_outlier_removal(raw, config)
    smoothed = apply_smoothing(cleaned, config)
    mean, std = smoothed.mean(axis=0, keepdims=True), smoothed.std(axis=0, keepdims=True)
    normalized = (smoothed - mean) / (std + 1e-6)

    rows = []
    for sub in SUBCARRIERS_SHOWN:
        for t in range(raw.shape[0]):
            rows.append({
                "packet_idx": t, "subcarrier": sub, "raw": raw[t, sub], "cleaned": cleaned[t, sub],
                "smoothed": smoothed[t, sub], "normalized": normalized[t, sub],
            })
    return pd.DataFrame(rows)


def plot_sample(df: pd.DataFrame, session_dir_rel: str, experiment_name: str, out_path) -> None:
    sub = SUBCARRIERS_SHOWN[0]
    d = df[df["subcarrier"] == sub]
    stages = ["raw", "cleaned", "smoothed", "normalized"]

    fig, axes = plt.subplots(len(stages), 1, figsize=(9, 8), sharex=True)
    for ax, stage in zip(axes, stages):
        ax.plot(d["packet_idx"], d[stage], color=ACCENT, linewidth=1)
        ax.set_ylabel(stage)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("packet index")
    fig.suptitle(f"{experiment_name} -- {session_dir_rel} (subcarrier {sub})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    manifest = build_ch6_manifest_all_labels()
    authorized = manifest[manifest["label"] == "authorized"].sort_values("date", ascending=False)
    session_dir_rel = authorized.iloc[0]["session_dir"]
    print(f"representative session: {session_dir_rel}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for experiment_name in SAMPLE_EXPERIMENTS:
        df = build_sample(session_dir_rel, experiment_name)
        csv_path = OUT_DIR / f"{experiment_name}_sample.csv"
        png_path = OUT_DIR / f"{experiment_name}_sample.png"
        df.to_csv(csv_path, index=False)
        plot_sample(df, session_dir_rel, experiment_name, png_path)
        print(f"  {experiment_name}: wrote {csv_path.name}, {png_path.name}")


if __name__ == "__main__":
    main()
