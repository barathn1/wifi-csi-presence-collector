# ML training pipeline (Day-1 POC)

Status: **Day 1 data only** (`data/authorized/2026-09-09`, `data/unauthorized/2026-09-09`,
`data/none/2026-09-09`). Every accuracy number in this pipeline is a same-day, session/person-disjoint
number -- NOT a cross-day number. Cross-day backtesting (train on Day 1+2, test on held-out Day 3)
activates automatically in `data_pipeline/splits.py::day_disjoint_split` once more dates exist under
`data/`, but there is nothing to hold out yet. See `reports/day1_findings.md` for the actual numbers
and what they mean; this file is just "how to run it."

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt
```

`MPLBACKEND=Agg` is appended to `.venv/bin/activate` -- matplotlib's default Tk backend crashes when
scripts run headless/non-interactively (hit this the first time `feature_importance.py` ran).

## Data does not leave `data/` unmodified

Nothing in this pipeline writes into `data/`. Every derived artifact (decoded per-session arrays,
window indices, feature matrices, figures) goes under `ml/data_pipeline/cache/` or
`ml/visualization/figures/`, both gitignored. Ad-hoc scratch/debug output goes in the repo-root `temp/`
folder (also gitignored) -- delete it freely.

## Pipeline order

```bash
# 1. Decode + window: builds per-session .npz caches and a window index (one 1s/200-packet window
#    per row, 50% overlap). Filters to the dominant csi_len=372 (186-subcarrier) frame type -- see
#    windowing.py's docstring for why.
python3 -m ml.data_pipeline.windowing --window-packets 200

# 2. Handcrafted statistical features (mean/std/skew/kurtosis per subcarrier x amplitude/phase),
#    both raw and Variant-A empty-room-calibrated -- input to the classical baseline AND to the
#    visualization scripts.
python3 -m ml.data_pipeline.features

# 3. Stage 1: RandomForest x 4 tasks x {raw, calibA}, full session/person-disjoint 5-fold CV,
#    naive-random-split logged alongside for comparison. Cheapest path to a real number.
python3 -m ml.training.run_stage1

# 4. Stage 2: the transformer/attention model zoo x tasks 0/A/B, single 80/20 session-disjoint split
#    (CPU-only budget -- see train.py's docstring).
python3 -m ml.training.run_stage2 --epochs 3

# 4b. taskD (authorized vs everything else) gets dedicated scripts, not run_stage2 -- it's the central
#     question, and a blended accuracy hides the auth/unauthorized-vs-none false-accept breakdown that
#     actually matters (see reports/day1_findings.md):
python3 -m ml.training.run_taskD_focus --epochs 4              # single split, false-accept breakdown
python3 -m ml.training.run_taskD_crossattn_cv --epochs 4        # same, full CV (use this before trusting any number -- a single-split/single-seed run on this task swung from AUROC 0.757 to 0.698 on re-run)
python3 -m ml.training.run_taskD_segment_aggregation --epochs 4 # does deciding from ~5s/~10s of averaged windows beat a single ~1s window?

# 5. Visualizations -- written to the top-level visualizations/ folder, NOT ml/visualization/figures/
#    (see ../visualizations/README.md for the full command list + what to look at first):
python3 -m ml.visualization.interactive_heatmap             # playable per-session heatmaps, all 37 sessions
python3 -m ml.visualization.signal_vs_noise --n-perm 5000    # noise floor via session-level permutation test
python3 -m ml.visualization.class_fingerprint
python3 -m ml.visualization.effect_size_heatmap
python3 -m ml.visualization.embedding_scatter
python3 -m ml.visualization.feature_importance
```

**Visualizations live in `visualizations/` at the repo root** (not buried under `ml/`) so they're easy
to find in a file browser -- `visualizations/static/*.png` and `visualizations/interactive/*.html`
(open `visualizations/interactive/index.html` to browse/filter all sessions). Both gitignored,
regenerable from the commands above.

**Neural-net results need a fixed seed AND multiple seeds, not just a fixed train/test split.**
`train.py::train_classifier` takes a `seed` argument (default 0) after discovering that re-training the
identical model on the identical fold swung AUROC from 0.757 to 0.698 and unauthorized-false-accept from
26% to 57% -- pure random-init/shuffling noise, not a real difference. Every single-run Stage 2 number
in `reports/day1_findings.md` should be read with this in mind until re-checked across seeds.

All results append to `ml/evaluation/results/experiment_log.csv` (one row per fold + a `mean` summary
row per task/preprocessing/model combo). Figures land in `ml/visualization/figures/`.

## The 5 tasks (`data_pipeline/tasks.py`)

`taskD` is the primary/production framing -- the other four are either easier sub-problems (task0) or
alternative cuts of the same underlying question (A/C). See `reports/day1_findings.md` for why a single
blended accuracy for D is misleading on its own (it hides the auth-vs-unauthorized false-accept rate
behind easy-to-reject empty rooms).

| Task | Question | Split used |
|---|---|---|
| `taskD_auth_vs_nonauth` | **authorized vs everything else** (unauthorized OR none) -- the actual production question | session-disjoint 5-fold; some folds hold out zero authorized sessions by chance (not label-stratified) -- skip those rather than let EER/AUROC crash on a single-class fold, see `run_taskD_focus.py::pick_valid_fold` |
| `task0_presence` | empty room vs occupied | session-disjoint 5-fold |
| `taskA_threeway` | authorized / unauthorized / none | session-disjoint 5-fold |
| `taskB_identity` | anjali vs barath (occupied-authorized only) | session-disjoint 5-fold -- **must** be session-, not person-disjoint: with only 2 identities, holding out a whole person leaves the other fold with zero examples of the class it needs to predict |
| `taskC_openset_proxy` | authorized vs unauthorized only (excludes `none`), closed-set proxy for real open-set verification | leave-one-unauthorized-person-out, with a fixed held-out slice of authorized sessions too (both classes must appear in every test fold or EER/AUROC are undefined) |

## The model zoo (`models/`)

- `baselines.py` -- RandomForest on handcrafted stats. **Note**: tree models are invariant to any
  per-feature monotonic transform, so raw vs Variant-A-calibrated will always score identically here by
  construction -- the calibration ablation only means something on the neural models below.
- `cnn1d.py`, `lstm.py` -- cheap deep-learning floors (matches the ESP32 paper's own CNN-only ablation).
- `transformer_whofi.py` -- WhoFi replica: single amplitude-only branch, L2-normalized embedding.
- `transformer_dualbranch.py` -- ESP32 paper replica (99.82% on 6 subjects): separate amplitude/phase
  branches, late-fusion concat.
- `transformer_crossattn.py` -- **own architecture**: amplitude and phase branches bidirectionally
  cross-attend to each other instead of only meeting at a final concat.
- `transformer_calib_context.py` -- **own architecture**: the day's empty-room baseline stats are fed
  in as tokens the main sequence cross-attends to, instead of being subtracted as fixed preprocessing.
- `cascade.py` -- **own idea, systems-level**: a cheap presence-gate in front of the harder
  identity/verification model, so the harder model is never asked to guess on an empty room.

## Decision horizon: windows are already sequences, not single packets (`evaluation/segment_aggregation.py`)

No model anywhere in this pipeline ever sees one packet in isolation -- the smallest unit is already a
200-packet (~1s) window (`windowing.py`). `segment_aggregation.py` goes further: it averages a trained
model's score over several consecutive overlapping windows (10 windows =~5s, 20 =~10s) before deciding,
matching ARGUS's (arXiv:2608.14670) finding that this kind of segment-level aggregation is a "free"
accuracy gain over a single-window decision, with zero retraining. See `run_taskD_segment_aggregation.py`
for the auth-vs-non-auth version of this comparison.

## Calibration (`data_pipeline/calibration.py`)

- **Variant A**: standardize a day's windows against that same day's `none`-session stats (OpenCSI-style).
- **Variant B**: re-reference today's signal onto a *different* day's empty-room stats -- the literal
  "calibrate today, normalize to a prior day's baseline" mechanism. Can't be validated for real cross-day
  benefit until Day 2 exists; `day1_proxy_baselines()` exercises the code path today using the two
  Day-1 `none` sessions furthest apart in time as a mechanical stand-in.

## When Day 2 / Day 3 land

1. Re-run `windowing.py` and `features.py` (they read the whole `data/manifest.csv`, so new dates are
   picked up automatically).
2. `splits.py::day_disjoint_split` will stop returning `None` -- switch `run_stage1.py`/`run_stage2.py`
   to use it instead of the k-folds once there are >=2 dates.
3. Re-run the calibration ablation (Variant A and B) for real -- this is the point where the empty-room
   calibration hypothesis actually gets tested, not just mechanically exercised.
4. Day 3, once collected, is held out entirely until the final report -- never pooled into training/CV.
