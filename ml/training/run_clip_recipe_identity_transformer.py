"""Same clip-recipe experiment as run_clip_recipe_identity.py (identical cleaning, 3s/1s/50%-coverage
clips, latest-session-per-day holdout, cross-day convention) but with the model swapped from
RandomForest to a Transformer, per direct request. The recipe only defines amplitude (no phase), so
this uses WhoFiTransformer (models/transformer_whofi.py, amplitude-only -- the actual "WhoFi" model,
not this repo's amplitude+phase "dualbranch" replica used elsewhere in this investigation).

Clips have a variable packet count (real 3s time windows over a jittery real packet rate) -- resampled
per clip to a fixed length (day's own average rate x 3s) via linear interpolation, the same resampling
already used for the Wi-Gait gait-cycle approximation, so they can be batched.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT, load_session
from ml.data_pipeline.windowing import load_manifest
from ml.data_pipeline.wigait_approx_preprocessing import resample_cycle
from ml.visualization._preprocess_empty_room import clean_packet_mask, drop_duplicate_timestamps
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import train_classifier, log_rows
from ml.training.run_clip_recipe_identity import TRAIN_PEOPLE, CLIP_LEN_S, STRIDE_S, COVERAGE_THRESHOLD

EPOCHS = 15
SEED = 0


def clip_sequences_for_session(session_dir_rel: str, meta: dict, clip_len_samples: int) -> list[dict]:
    session = load_session(REPO_ROOT / "data" / session_dir_rel)
    npz = session.npz

    clean = clean_packet_mask(npz)
    clean_idx = np.flatnonzero(clean)
    if len(clean_idx) == 0:
        return []
    clean_idx, clean_times = drop_duplicate_timestamps(clean_idx, npz["device_time_us"][clean_idx].astype(np.int64))

    t0 = clean_times.min()
    t_all_s = (clean_times - t0) / 1e6
    duration_s = float(t_all_s.max())

    csi_len_clean = npz["csi_len"][clean_idx].astype(np.int64)
    lens, counts = np.unique(csi_len_clean, return_counts=True)
    dom_len = int(lens[np.argmax(counts)])
    dom_sel = csi_len_clean == dom_len
    dom_idx = clean_idx[dom_sel]
    dom_t_s = t_all_s[dom_sel]

    csi_offset = npz["csi_offset"].astype(np.int64)
    csi_flat = npz["csi_flat"]
    n_sub = dom_len // 2
    offs = csi_offset[dom_idx]
    gather = offs[:, None] + np.arange(dom_len)[None, :]
    raw = csi_flat[gather].astype(np.float32)
    dom_amp = np.hypot(raw[:, 1::2], raw[:, 0::2])

    expected_rate = float(meta.get("avg_rate_hz") or (len(npz["csi_len"]) / meta["duration_s"]))
    min_required = COVERAGE_THRESHOLD * expected_rate * CLIP_LEN_S

    n_clips = max(0, int(np.floor((duration_s - CLIP_LEN_S) / STRIDE_S)) + 1)
    clips = []
    for i in range(n_clips):
        start, end = i * STRIDE_S, i * STRIDE_S + CLIP_LEN_S
        lo_all = np.searchsorted(t_all_s, start, side="left")
        hi_all = np.searchsorted(t_all_s, end, side="left")
        if hi_all - lo_all < min_required:
            continue
        lo_d = np.searchsorted(dom_t_s, start, side="left")
        hi_d = np.searchsorted(dom_t_s, end, side="left")
        if hi_d - lo_d < 3:
            continue
        resampled = resample_cycle(dom_amp[lo_d:hi_d], target_len=clip_len_samples)
        clips.append({"amp": resampled.astype(np.float32), "motion": meta.get("motion") or "none",
                       "start_s": start, "n_sub": n_sub})
    return clips


def clip_len_for_day(manifest: pd.DataFrame, day: str) -> int:
    day_auth = manifest[(manifest["label"] == "authorized") & manifest["person_id"].isin(TRAIN_PEOPLE)
                         & manifest["session_dir"].str.contains(f"/{day}/", regex=False)]
    avg_rate = (day_auth["sample_count"].astype(float) / day_auth["duration_s"].astype(float)).mean()
    return max(1, round(avg_rate * CLIP_LEN_S))


def build_clip_table(motion_filter: str | None) -> pd.DataFrame:
    manifest = load_manifest()
    auth = manifest[manifest["label"] == "authorized"].copy()
    auth = auth[auth["person_id"].isin(TRAIN_PEOPLE)]
    if motion_filter is not None:
        auth = auth[auth["motion"] == motion_filter]

    clip_len_by_day = {d: clip_len_for_day(manifest, d) for d in sorted(auth["session_dir"].str.split("/").str[1].unique())}

    rows = []
    for _, row in auth.iterrows():
        date = row["session_dir"].split("/")[1]
        clips = clip_sequences_for_session(row["session_dir"], row.to_dict(), clip_len_by_day[date])
        for c in clips:
            rows.append({
                "person_id": row["person_id"], "motion": c["motion"], "date": date,
                "session_dir": row["session_dir"], "start_ts": row["start_ts"] + c["start_s"],
                "n_sub": c["n_sub"], "amp": c["amp"],
            })
    return pd.DataFrame(rows)


def latest_session_per_person_day(clip_table: pd.DataFrame) -> set[str]:
    held_out = set()
    for (person, date), grp in clip_table.groupby(["person_id", "date"]):
        latest_start = grp.groupby("session_dir")["start_ts"].min().sort_values()
        held_out.add(latest_start.index[-1])
    return held_out


class ClipDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, classes: list[str], n_target: int):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        amps = []
        for amp, n_sub in zip(rows["amp"], rows["n_sub"]):
            if n_sub < n_target:
                amp = np.concatenate([amp, np.zeros((amp.shape[0], n_target - n_sub), dtype=np.float32)], axis=1)
            amps.append(amp)
        self.amp = np.stack(amps)
        self.labels = rows["person_id"].map(self.class_to_idx).values

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        amp_t = torch.from_numpy(self.amp[i])
        return amp_t, amp_t, int(self.labels[i])  # phase slot unused by WhoFiTransformer


def anjali_recall(model, ds: ClipDataset, classes: list[str]) -> float:
    anjali_idx = classes.index("anjali")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for amp, phase, label in DataLoader(ds, batch_size=64):
            mask = label == anjali_idx
            if not mask.any():
                continue
            pred = model(amp, phase).argmax(dim=-1)
            correct += (pred[mask] == label[mask]).sum().item()
            total += mask.sum().item()
    return correct / total if total else float("nan")


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, n_target: int) -> dict:
    classes = sorted(set(train_df["person_id"]) | set(test_df["person_id"]))
    train_ds = ClipDataset(train_df, classes, n_target)
    test_ds = ClipDataset(test_df, classes, n_target)

    torch.manual_seed(SEED)
    model = WhoFiTransformer(n_target, len(classes))
    result = train_classifier(model, train_ds, test_ds, epochs=EPOCHS, seed=SEED)
    recall = anjali_recall(model, test_ds, classes)
    return {"accuracy": result["accuracy"], "anjali_recall": recall, "n_train": len(train_ds), "n_test": len(test_ds)}


def run_same_day(clip_table: pd.DataFrame, held_out: set[str], motion_tag: str) -> list[dict]:
    results = []
    for date, day_grp in clip_table.groupby("date"):
        test_mask = day_grp["session_dir"].isin(held_out)
        train_df, test_df = day_grp[~test_mask], day_grp[test_mask]
        if train_df.empty or test_df.empty or train_df["person_id"].nunique() < 2:
            print(f"  {date} ({motion_tag}): skipped, train={len(train_df)} test={len(test_df)} "
                  f"train_classes={train_df['person_id'].nunique() if not train_df.empty else 0}")
            continue
        n_target = int(day_grp["n_sub"].max())
        r = train_and_eval(train_df, test_df, n_target)
        print(f"  {date} ({motion_tag}): train={r['n_train']} test={r['n_test']} "
              f"acc={r['accuracy']*100:.1f}% anjali_recall={r['anjali_recall']*100:.1f}%")
        results.append({"day": date, "motion": motion_tag, "split": "same_day", **r})
    return results


def run_cross_day(clip_table: pd.DataFrame, motion_tag: str) -> dict | None:
    dates = sorted(clip_table["date"].unique())
    if len(dates) < 2:
        return None
    train_date, test_date = dates[0], dates[-1]
    train_df = clip_table[clip_table["date"] == train_date]
    test_df = clip_table[clip_table["date"] == test_date]
    if train_df.empty or test_df.empty:
        print(f"  cross-day ({motion_tag}): skipped, empty train or test")
        return None

    # z-score amplitude per subcarrier against each day's OWN clip population (see
    # run_clip_recipe_identity.zscore_against_own_day for the same reasoning), then pad the
    # shorter day up to the longer day's subcarrier count.
    n_target = int(max(train_df["n_sub"].max(), test_df["n_sub"].max()))

    def zscore_group(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        stacked = np.stack(df["amp"].to_list())  # (N, T, n_sub)
        mu = stacked.mean(axis=(0, 1))
        sigma = stacked.std(axis=(0, 1)) + 1e-6
        df["amp"] = [((a - mu) / sigma).astype(np.float32) for a in df["amp"]]
        return df

    train_df, test_df = zscore_group(train_df), zscore_group(test_df)
    r = train_and_eval(train_df, test_df, n_target)
    print(f"  cross-day ({motion_tag}): train={train_date} ({r['n_train']}) -> test={test_date} "
          f"({r['n_test']}) acc={r['accuracy']*100:.1f}% anjali_recall={r['anjali_recall']*100:.1f}%")
    return {"day": f"{train_date}->{test_date}", "motion": motion_tag, "split": "cross_day", **r}


def main() -> None:
    all_results = []
    for motion_tag, motion_filter in [("pooled", None), ("standing", "standing"), ("walking", "walking")]:
        print(f"\n=== motion filter: {motion_tag} (transformer) ===")
        clip_table = build_clip_table(motion_filter)
        if clip_table.empty:
            print("  no clips at all, skipping")
            continue
        print(f"  total clips: {len(clip_table)}")

        held_out = latest_session_per_person_day(clip_table)
        all_results += run_same_day(clip_table, held_out, motion_tag)
        cross = run_cross_day(clip_table, motion_tag)
        if cross:
            all_results.append(cross)

    print("\n=== clip-recipe identity results (WhoFiTransformer, anjali vs barath) ===")
    print(f"{'split':<10}{'day':<24}{'motion':<10}{'accuracy':>10}{'anjali_recall':>16}")
    for r in all_results:
        print(f"{r['split']:<10}{r['day']:<24}{r['motion']:<10}{r['accuracy']*100:>9.1f}%{r['anjali_recall']*100:>15.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": f"clip_recipe_transformer_{r['split']}",
        "task": "taskB_identity", "preprocessing": "clip_recipe_3s_1s_50pct", "model": "whofi_transformer",
        "split_type": f"{r['split']}_latest_session_holdout", "fold": 0, "accuracy": r["accuracy"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; day={r['day']}; motion={r['motion']}",
    } for r in all_results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
