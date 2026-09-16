"""Scores a live-collected session against the deployed home model (train_home_model.py).

    python3 -m ml.inference.score_live_session data/authorized/2026-09-15/<session_dir> [--true-person anjali]

Runs the identical clip recipe used at training time (3s/1s-stride clips, drop packets that are
incomplete/corrupted/duplicate-timestamp, drop clips with <50% expected packet coverage, mean/std/
skew/kurt per subcarrier), pads to the model's trained width if this session's native subcarrier
count is narrower, and prints a per-clip prediction plus a session-level majority vote. If
--true-person is given (you know who was actually being recorded), also prints per-clip and
session-level accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.training.run_clip_recipe_identity import clip_features_for_session, stack_features

ARTIFACT_PATH = REPO_ROOT / "ml" / "models_out" / "home_model_v2.joblib"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session_dir", help="path to a live session dir (containing samples.npz + metadata.json)")
    p.add_argument("--true-person", default=None, help="ground truth, if known, to score accuracy")
    p.add_argument("--artifact", default=str(ARTIFACT_PATH))
    args = p.parse_args()

    art = joblib.load(args.artifact)
    model, n_target = art["model"], art["n_target"]
    print(f"loaded {args.artifact} (trained on {art['trained_on']}, created {art['created']})")
    print(f"NOTE: {art['notes']}\n")

    session_dir = Path(args.session_dir).resolve()
    meta_path = session_dir / "metadata.json"
    import json
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    data_root = (REPO_ROOT / "data").resolve()
    session_dir_rel = str(session_dir.relative_to(data_root)).replace("\\", "/")

    clips = clip_features_for_session(session_dir_rel, meta)
    if not clips:
        print("no usable clips (session too short, or every clip failed the coverage check)")
        return

    n_sub = clips[0]["n_sub"]
    if n_sub != n_target:
        print(f"session native subcarriers={n_sub}, training width={n_target} -> zero-padding")

    import pandas as pd
    df = pd.DataFrame({"features": [c["features"] for c in clips], "n_sub": [c["n_sub"] for c in clips]})
    X = stack_features(df, n_target)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    print(f"{len(clips)} clips scored:")
    for i, (p_label, prob) in enumerate(zip(pred, proba)):
        conf = prob[classes.index(p_label)]
        print(f"  clip {i:3d} (t~{clips[i]['start_s']:.0f}s): predicted={p_label:<8} confidence={conf:.2f}")

    vals, counts = np.unique(pred, return_counts=True)
    majority = vals[np.argmax(counts)]
    print(f"\nmajority vote over {len(clips)} clips: {majority} "
          f"({dict(zip(vals.tolist(), counts.tolist()))})")

    if args.true_person:
        clip_acc = (pred == args.true_person).mean()
        print(f"\nground truth={args.true_person}")
        print(f"  per-clip accuracy: {clip_acc*100:.1f}% ({(pred == args.true_person).sum()}/{len(pred)})")
        print(f"  session-level (majority vote) correct: {majority == args.true_person}")


if __name__ == "__main__":
    main()
