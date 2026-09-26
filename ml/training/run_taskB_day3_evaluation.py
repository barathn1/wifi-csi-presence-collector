"""Day 3 (2026-09-15) was captured on channel 6 / 20MHz (HT20), vs. Day 1's channel 11 / 40MHz
(HT40, native 186 subcarriers) and Day 2's channel 11 / 20MHz (HT20, native 128 subcarriers) --
see run_taskB_day_to_day.py's docstring for how the HT20/HT40 split was established. Checked with
`decode_csi.py --buckets` on a Day 3 session here: Day 3's dominant bucket is 128 subcarriers,
96.4% of packets -- i.e. Day 3 is in the SAME native capture mode as Day 2 (channel NUMBER differs,
but channel WIDTH/mode, which is what determines subcarrier count, matches Day 2 not Day 1).

Two questions, using the clip recipe (run_clip_recipe_identity.py: 3s/1s-stride clips, drop
<50%-coverage, mean/std/skew/kurt per subcarrier, RandomForest):

1. Is Day 3 good for identification on its own? Same-day session-disjoint holdout restricted to
   Day 3 (latest session per person held out, rest of Day 3 trains) -- the same methodology already
   used for Day 1 and Day 2 in run_clip_recipe_identity.py, so directly comparable.
2. Train on Day 1 + Day 2 pooled, test on Day 3 (the actual home-model deployment scenario --
   train_home_model.py trains this same pooled model but doesn't score it against held-out Day 3
   ground truth; this does). Day 1/2/3 rows are z-scored against their OWN day's clip population
   (zscore_against_own_day) before zero-padding to the shared 186-wide target, same convention as
   run_clip_recipe_identity.py's cross_day path.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.training.run_clip_recipe_identity import (
    TRAIN_PEOPLE, build_clip_table, latest_session_per_person_day, run_same_day,
    stack_features, zscore_against_own_day,
)
from ml.training.train import log_rows

DAY1, DAY2, DAY3 = "2026-09-09", "2026-09-10", "2026-09-15"
N_TARGET = 186  # Day 1's native width, same convention as train_home_model.py


def run_cross_day_pooled(clip_table, train_dates: list[str], test_date: str, motion_tag: str) -> dict | None:
    train_df = clip_table[clip_table["date"].isin(train_dates)]
    test_df = clip_table[clip_table["date"] == test_date]
    if train_df.empty or test_df.empty:
        print(f"  cross-day pooled ({motion_tag}): skipped, empty train or test")
        return None

    zs = zscore_against_own_day(clip_table[clip_table["date"].isin(train_dates + [test_date])])
    train_df = zs.loc[train_df.index]
    test_df = zs.loc[test_df.index]

    X_train = stack_features(train_df, N_TARGET)
    X_test = stack_features(test_df, N_TARGET)
    y_train, y_test = train_df["person_id"].values, test_df["person_id"].values

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    anjali_mask = y_test == "anjali"
    barath_mask = y_test == "barath"
    anjali_recall = accuracy_score(y_test[anjali_mask], pred[anjali_mask]) if anjali_mask.any() else float("nan")
    barath_recall = accuracy_score(y_test[barath_mask], pred[barath_mask]) if barath_mask.any() else float("nan")
    print(f"  cross-day pooled ({motion_tag}): train={'+'.join(train_dates)} ({len(train_df)}) -> "
          f"test={test_date} ({len(test_df)}) acc={acc*100:.1f}% "
          f"anjali_recall={anjali_recall*100:.1f}% barath_recall={barath_recall*100:.1f}%")
    return {"day": f"{'+'.join(train_dates)}->{test_date}", "motion": motion_tag, "split": "cross_day_pooled",
            "accuracy": acc, "anjali_recall": anjali_recall, "barath_recall": barath_recall,
            "n_train": len(train_df), "n_test": len(test_df)}


def main() -> None:
    all_results = []
    for motion_tag, motion_filter in [("pooled", None), ("standing", "standing"), ("walking", "walking")]:
        print(f"\n=== motion filter: {motion_tag} ===")
        clip_table = build_clip_table(motion_filter)
        if clip_table.empty:
            print("  no clips at all, skipping")
            continue
        dates_present = sorted(clip_table["date"].unique())
        print(f"  total clips: {len(clip_table)}, by person/date:")
        print(clip_table.groupby(["date", "person_id"]).size())
        if DAY3 not in dates_present:
            print(f"  {DAY3} not present for this motion filter, skipping")
            continue

        # Q1: is Day 3 good for identification on its own?
        day3_only = clip_table[clip_table["date"] == DAY3]
        held_out = latest_session_per_person_day(day3_only)
        print(f"  Day3 held-out (latest-per-person) sessions: {sorted(held_out)}")
        all_results += run_same_day(day3_only, held_out, motion_tag)

        # Q2: train Day1+Day2 pooled -> test Day3
        if DAY1 in dates_present and DAY2 in dates_present:
            cross = run_cross_day_pooled(clip_table, [DAY1, DAY2], DAY3, motion_tag)
            if cross:
                all_results.append(cross)
        else:
            print(f"  {DAY1} or {DAY2} missing for this motion filter, skipping cross-day pooled")

    print("\n=== Day 3 evaluation results (RandomForest, anjali vs barath) ===")
    print(f"{'split':<18}{'day':<24}{'motion':<10}{'accuracy':>10}{'anjali_recall':>16}{'barath_recall':>16}")
    for r in all_results:
        print(f"{r['split']:<18}{r['day']:<24}{r['motion']:<10}{r['accuracy']*100:>9.1f}%"
              f"{r['anjali_recall']*100:>15.1f}%{r.get('barath_recall', float('nan'))*100:>15.1f}%")

    log_rows([{
        "timestamp": datetime.now(timezone.utc).isoformat(), "stage": f"day3_{r['split']}",
        "task": "taskB_identity", "preprocessing": "clip_recipe_3s_1s_50pct", "model": "random_forest",
        "split_type": f"{r['split']}_latest_session_holdout" if r["split"] == "same_day" else r["split"],
        "fold": 0, "accuracy": r["accuracy"], "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": (f"anjali_recall={r['anjali_recall']:.4f}; "
                  f"barath_recall={r.get('barath_recall', float('nan')):.4f}; "
                  f"day={r['day']}; motion={r['motion']}"),
    } for r in all_results])
    print("\nlogged to", REPO_ROOT / "ml/evaluation/results/experiment_log.csv")


if __name__ == "__main__":
    main()
