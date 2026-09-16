"""anjali-vs-barath identification using the exact recipe:
  1. amplitude = sqrt(real^2+imag^2) per subcarrier per packet.
  2. drop incomplete/corrupted (first_word_invalid)/duplicate-timestamp packets.
  3. 3s clips, 1s stride (overlapping).
  4. drop clips with <50% of the session's expected packet count.
  5. label each clip with activity (motion) and person.
  Train: anjali + barath only (no blind strangers this run).
  Split: for each person, each day, hold out their chronologically LATEST session for
  same-day ("day-to-day") testing; train on that day's other sessions.
  Cross-day: train on ALL of Day1, test on ALL of Day2 (matches this project's established
  cross-day convention throughout this investigation).

Reuses the packet-cleaning primitives from visualization/_preprocess_empty_room.py (this recipe
was first built there, for empty-room data; this generalizes it to authorized/occupied sessions)
and the mean/std/skew/kurt-per-subcarrier feature convention from data_pipeline/features.py, since
clips have a variable packet count and this reduction is the natural fixed-length representation --
avoids needing to resample every clip to a common sequence length for a classifier.

Day 1 is native 186 subcarriers, Day 2 is native 128 (HT40 vs HT20, established earlier in this
investigation) -- cross-day features are z-scored against each day's own empty-room baseline (so
the fill for Day2's 58 missing dimensions can be a principled 0 = "at baseline", not an arbitrary
raw value) then zero-padded to 186. This doesn't change RandomForest's OWN 128 real dimensions at
all (RF splits are invariant to a per-feature affine transform, verified earlier in this
investigation) -- it only makes the padding well-defined.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ml.data_pipeline.decode_csi import REPO_ROOT, load_session
from ml.data_pipeline.windowing import load_manifest
from ml.data_pipeline.features import _moments
from ml.visualization._preprocess_empty_room import clean_packet_mask, drop_duplicate_timestamps
from ml.training.train import log_rows

TRAIN_PEOPLE = ["anjali", "barath"]
CLIP_LEN_S = 3.0
STRIDE_S = 1.0
COVERAGE_THRESHOLD = 0.5


def clip_features_for_session(session_dir_rel: str, meta: dict) -> list[dict]:
    """One row per kept clip: {features (4*n_sub,), motion, start_s}."""
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
        if hi_d <= lo_d:
            continue
        mean, std, skew, kurt = _moments(dom_amp[lo_d:hi_d])
        feat = np.nan_to_num(np.concatenate([mean, std, skew, kurt]), nan=0.0).astype(np.float32)
        clips.append({"features": feat, "motion": meta.get("motion") or "none", "start_s": start,
                       "n_sub": n_sub})
    return clips


def build_clip_table(motion_filter: str | None) -> pd.DataFrame:
    manifest = load_manifest()
    auth = manifest[manifest["label"] == "authorized"].copy()
    auth = auth[auth["person_id"].isin(TRAIN_PEOPLE)]
    if motion_filter is not None:
        auth = auth[auth["motion"] == motion_filter]

    rows = []
    for _, row in auth.iterrows():
        meta = row.to_dict()
        clips = clip_features_for_session(row["session_dir"], meta)
        date = row["session_dir"].split("/")[1]
        for c in clips:
            rows.append({
                "person_id": row["person_id"], "motion": c["motion"], "date": date,
                "session_dir": row["session_dir"], "start_ts": row["start_ts"] + c["start_s"],
                "n_sub": c["n_sub"], "features": c["features"],
            })
    return pd.DataFrame(rows)


def latest_session_per_person_day(clip_table: pd.DataFrame) -> set[str]:
    """The chronologically-latest session_dir per (person, date) -- held out from training."""
    held_out = set()
    for (person, date), grp in clip_table.groupby(["person_id", "date"]):
        latest_start = grp.groupby("session_dir")["start_ts"].min().sort_values()
        held_out.add(latest_start.index[-1])
    return held_out


def stack_features(rows: pd.DataFrame, n_target: int) -> np.ndarray:
    """rows' 'features' column has 4*n_sub entries; pads (mean/std/skew/kurt blocks
    independently) to 4*n_target with zeros if n_sub < n_target."""
    out = []
    for feat, n_sub in zip(rows["features"], rows["n_sub"]):
        if n_sub == n_target:
            out.append(feat)
            continue
        blocks = np.split(feat, 4)  # mean, std, skew, kurt, each (n_sub,)
        padded = [np.concatenate([b, np.zeros(n_target - n_sub, dtype=np.float32)]) for b in blocks]
        out.append(np.concatenate(padded))
    return np.stack(out)


def zscore_against_own_day(rows: pd.DataFrame) -> pd.DataFrame:
    """z-score each row's feature vector against the MEAN/STD OF THAT SAME DAY's own clip
    population (there's no empty-room analog restricted to these two people's occupied clips, so
    this self-normalizes each day rather than referencing a `none`-session baseline)."""
    rows = rows.copy()
    for date, grp in rows.groupby("date"):
        X = np.stack(grp["features"].to_list())
        mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-6
        for idx, feat in zip(grp.index, grp["features"]):
            rows.at[idx, "features"] = ((feat - mu) / sigma).astype(np.float32)
    return rows


def run_same_day(clip_table: pd.DataFrame, held_out: set[str], motion_tag: str) -> list[dict]:
    results = []
    for date, day_grp in clip_table.groupby("date"):
        test_mask = day_grp["session_dir"].isin(held_out)
        train_df, test_df = day_grp[~test_mask], day_grp[test_mask]
        if train_df.empty or test_df.empty:
            print(f"  {date} ({motion_tag}): skipped, empty train or test (train={len(train_df)}, test={len(test_df)})")
            continue
        n_sub = int(day_grp["n_sub"].iloc[0])
        X_train, X_test = stack_features(train_df, n_sub), stack_features(test_df, n_sub)
        y_train, y_test = train_df["person_id"].values, test_df["person_id"].values

        model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        anjali_mask = y_test == "anjali"
        anjali_recall = accuracy_score(y_test[anjali_mask], pred[anjali_mask]) if anjali_mask.any() else float("nan")
        print(f"  {date} ({motion_tag}): train={len(train_df)} test={len(test_df)} "
              f"acc={acc*100:.1f}% anjali_recall={anjali_recall*100:.1f}%")
        results.append({"day": date, "motion": motion_tag, "split": "same_day", "accuracy": acc,
                         "anjali_recall": anjali_recall, "n_train": len(train_df), "n_test": len(test_df)})
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

    zs = zscore_against_own_day(pd.concat([train_df, test_df]))
    train_df = zs.loc[train_df.index]
    test_df = zs.loc[test_df.index]

    n_target = int(train_df["n_sub"].max())
    X_train = stack_features(train_df, n_target)
    X_test = stack_features(test_df, n_target)
    y_train, y_test = train_df["person_id"].values, test_df["person_id"].values

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    anjali_mask = y_test == "anjali"
    anjali_recall = accuracy_score(y_test[anjali_mask], pred[anjali_mask]) if anjali_mask.any() else float("nan")
    print(f"  cross-day ({motion_tag}): train={train_date} ({len(train_df)}) -> test={test_date} "
          f"({len(test_df)}) acc={acc*100:.1f}% anjali_recall={anjali_recall*100:.1f}%")
    return {"day": f"{train_date}->{test_date}", "motion": motion_tag, "split": "cross_day", "accuracy": acc,
            "anjali_recall": anjali_recall, "n_train": len(train_df), "n_test": len(test_df)}


def main() -> None:
    all_results = []
    for motion_tag, motion_filter in [("pooled", None), ("standing", "standing"), ("walking", "walking")]:
        print(f"\n=== motion filter: {motion_tag} ===")
        clip_table = build_clip_table(motion_filter)
        if clip_table.empty:
            print("  no clips at all, skipping")
            continue
        print(f"  total clips: {len(clip_table)}, by person/date:")
        print(clip_table.groupby(["date", "person_id"]).size())

        held_out = latest_session_per_person_day(clip_table)
        print(f"  held-out (latest-per-person-per-day) sessions: {sorted(held_out)}")

        all_results += run_same_day(clip_table, held_out, motion_tag)
        cross = run_cross_day(clip_table, motion_tag)
        if cross:
            all_results.append(cross)

    print("\n=== clip-recipe identity results (RandomForest, anjali vs barath) ===")
    print(f"{'split':<10}{'day':<24}{'motion':<10}{'accuracy':>10}{'anjali_recall':>16}")
    for r in all_results:
        print(f"{r['split']:<10}{r['day']:<24}{r['motion']:<10}{r['accuracy']*100:>9.1f}%{r['anjali_recall']*100:>15.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": f"clip_recipe_{r['split']}",
        "task": "taskB_identity", "preprocessing": "clip_recipe_3s_1s_50pct", "model": "random_forest",
        "split_type": f"{r['split']}_latest_session_holdout", "fold": 0, "accuracy": r["accuracy"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"anjali_recall={r['anjali_recall']:.4f}; day={r['day']}; motion={r['motion']}",
    } for r in all_results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
