"""Part 6 -- environment control / positive-control sanity check.

If person-identity classification turns out to be weak (see baselines.py), the first question is
whether the pipeline/features are simply broken -- unable to pick up ANY real physical signal, person
or otherwise. This script answers that with a positive control: occupied vs empty-room ("presence")
classification, on the exact same features/splits/models as the person-ID task. Presence is a much
coarser, higher-SNR distinction (someone reflecting/absorbing signal at all vs an empty room) -- if this
task also failed, that would point at the pipeline; if it succeeds cleanly while person-ID does not,
that isolates the weak result to person-specific information specifically.

Also documents the environment-control limitations of this scope explicitly (see bottom of __main__):
receiver board and WiFi channel/bandwidth are already held fixed by construction (load_data.py's
channel-6, single-board restriction), and "same person, different position" cannot be tested here --
the collected metadata only distinguishes standing vs walking, not body position/orientation within
"standing".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from analysis.walking_ch6.baselines import MODELS, evaluate_fold, run_split_scheme, subsample_per_session
from analysis.walking_ch6.load_data import build
from ml.data_pipeline.splits import leave_one_day_out, session_disjoint_kfold

RESULTS_DIR = Path(__file__).resolve().parent / "cache"


def main():
    window_index, X = build()
    window_index, X = subsample_per_session(window_index, X)
    y = (window_index["identity"] != "empty_room").to_numpy(dtype=int)  # 1 = occupied, 0 = empty room
    print(f"presence task: {len(window_index)} windows, "
          f"{ (y==1).sum()} occupied / {(y==0).sum()} empty-room")

    rows = []
    rows += run_split_scheme(window_index, X, y, "presence_cross_session_5fold",
                              [(i, tr, te) for i, (tr, te) in
                               enumerate(session_disjoint_kfold(window_index, n_splits=5))],
                              "session_dir")
    rows += run_split_scheme(window_index, X, y, "presence_leave_one_day_out",
                              list(leave_one_day_out(window_index)), "session_dir")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "presence_control_results.csv", index=False)
    summary = df.groupby(["scheme", "model"])[["accuracy", "balanced_accuracy", "f1_macro"]].mean().round(3)
    print("\npresence (occupied vs empty room) -- positive control:")
    print(summary)
    print("\nThis is expected to score high. If it does, the feature/pipeline can clearly detect a real "
          "physical effect (presence) under the same strict splits used for person-ID -- so a weak "
          "person-ID result is not explained by a broken pipeline.")


if __name__ == "__main__":
    main()
