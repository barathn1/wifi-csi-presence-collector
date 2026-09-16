"""Trains the deployable "home model" -- anjali vs barath, RandomForest on the clip-recipe features
(run_clip_recipe_identity.py: 3s/1s-stride clips, drop <50%-coverage, mean/std/skew/kurt per
subcarrier) -- on Day 3 (2026-09-15) + Day 4 (2026-09-16) authorized data pooled (both motions,
since a live test won't necessarily control motion state), for live testing today (2026-09-16).

v2: earlier v1 (ml/models_out/home_model_v1.joblib, kept untouched) trained on Day1+Day2, padded to
186 subcarriers to bridge Day1's 186-sub/HT40 vs Day2's 128-sub/HT20 native modes, and its Day1->Day2
cross-day proxy accuracy ranged ~45-64% -- see that file's git history / docstring. Day 3 and Day 4
are BOTH native 128 subcarriers (channel 6, 20MHz/HT20 -- see run_taskB_day3_evaluation.py), so this
version needs no cross-mode padding: N_TARGET=128 matches both training days exactly, and should also
match whatever a same-setup live session records today.

RandomForest chosen because it was the most robust model across every cross-day comparison run in
this investigation (crop-128, Variant-B, paper-recipe+Variant-A, clip-recipe -- RF matched or beat
every deep-model variant tried, and never fully collapsed to a single class the way the transformer
did on several runs).

Expectations, stated plainly before you test live: this project has never seen day-to-day accuracy
above ~70-80% (see experiment_log.csv's day3_* and tfmamba_approx_day3_* entries: Day1+Day2->Day3
ranged 43-69% depending on method/motion) -- treat today's live test as a measurement, not a demo,
even though Day 3->Day 4 wasn't itself cross-day-validated before deploying this artifact.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.training.run_clip_recipe_identity import build_clip_table, stack_features, TRAIN_PEOPLE

ARTIFACT_DIR = REPO_ROOT / "ml" / "models_out"
ARTIFACT_PATH = ARTIFACT_DIR / "home_model_v2.joblib"
TRAIN_DATES = ["2026-09-15", "2026-09-16"]
N_TARGET = 128  # Day 3 and Day 4's shared native width -- see module docstring


def main() -> None:
    print(f"building clip table: {'+'.join(TRAIN_DATES)}, both motions pooled, anjali + barath...")
    clip_table = build_clip_table(motion_filter=None)
    clip_table = clip_table[clip_table["date"].isin(TRAIN_DATES)]
    print(f"total clips: {len(clip_table)}")
    print(clip_table.groupby(["date", "person_id"]).size())

    X = stack_features(clip_table, N_TARGET)
    y = clip_table["person_id"].values

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X, y)
    train_acc = model.score(X, y)
    print(f"\ntrain-set fit accuracy (NOT a generalization estimate, just a sanity check): {train_acc*100:.1f}%")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "n_target": N_TARGET,
        "classes": sorted(TRAIN_PEOPLE),
        "clip_len_s": 3.0,
        "stride_s": 1.0,
        "coverage_threshold": 0.5,
        "trained_on": TRAIN_DATES,
        "created": datetime.now(timezone.utc).isoformat(),
        "notes": ("Trained on Day3+Day4 (both native 128-sub/channel6/20MHz, no cross-mode padding "
                  "needed). This project's best day-to-day accuracy so far has been ~70-80% "
                  "same-day and ~43-69% cross-day -- treat live results as a measurement, not a "
                  "validated detector. See ml/evaluation/results/experiment_log.csv (day3_* and "
                  "tfmamba_approx_day3_* rows) for the full history."),
    }, ARTIFACT_PATH)
    print(f"\nwrote {ARTIFACT_PATH} ({ARTIFACT_PATH.stat().st_size / 1024:.0f} KB)")

    summary_path = ARTIFACT_DIR / "home_model_v2_training_summary.json"
    summary_path.write_text(json.dumps({
        "n_clips_total": len(clip_table),
        "clips_by_day_person": {f"{d}/{p}": int(n) for (d, p), n in
                                 clip_table.groupby(["date", "person_id"]).size().items()},
        "train_fit_accuracy": train_acc,
    }, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
