# ml_2 plan: channel-6-only, GPU-trained model battery

## Scope confirmed from the data itself (not assumed)

- **Channel filter**: `config_snapshot.hotspot.channel` in `metadata.json` is a red herring (always
  `1` on every date -- it's a serial/hotspot config value, not the negotiated WiFi channel). The real
  channel is per-packet (`channel_primary` in `samples.npz`, read via
  `ml.data_pipeline.decode_csi.channel_width_summary`). `ml/training/train_day3_ch6_model.py` already
  confirmed and hardcoded the answer: **2026-09-09 and 2026-09-10 are a different channel/band and are
  excluded; 2026-09-15, 09-16, 09-17, 09-21, 09-22 are all confirmed channel 6** (`TRAIN_DATES` /
  `TARGET_CHANNEL = 6` there). ml_2 reuses that exact filter rather than re-deriving or guessing dates.
- **Extra ESPs on the last 2 days**: 2026-09-21 and 2026-09-22 each recorded 3 ESP32 receivers in
  parallel (MACs `a4:cb:8f:d4:52:b0`, `ac:27:6e:a2:f2:78`, `ac:27:6e:a5:5b:c8`), confirmed via
  per-session file prefixes. `ml/models/receiver_fusion.py` + `ml/training/compare_receiver_fusion.py`
  already implement and evaluate a weight-shared 3-receiver fusion transformer for exactly those 2
  dates. ml_2 reuses that model, retrained on GPU, rather than rewriting fusion from scratch.
- **Labels**: `authorized` / `unauthorized` / `none` (empty room) directories, exactly as the existing
  `ml/data_pipeline/tasks.py` task definitions assume.

## What already exists in `ml/` and is reused as-is (not rewritten)

- `ml.data_pipeline.decode_csi` / `windowing` / `csi_resample` / `time_resample` -- raw CSI decode,
  per-session caching, time-normalized windowing.
- `ml.data_pipeline.calibration` -- Variant A (per-day empty-room self-calibration, per-subcarrier
  mean/std, not a global scalar) and Variant B (cross-day re-referencing). Used here via Variant A,
  same as `train_day3_ch6_model.py`.
- `ml.data_pipeline.splits` -- `leave_one_unauthorized_person_out` (the real open-set check: a stranger
  never seen in training) and `leave_one_day_out` (the real cross-day check) are exactly the two splits
  this project's own literature review flagged as the usually-unreported, actually-hard evaluations.
  ml_2 evaluates every model -- classical and deep -- under both, not just a single random split.
  `session_disjoint_kfold` is used for the closed-set sanity checks (task0/taskB/taskE).
  `naive_random_split` is reported alongside the honest split for at least one model so the leakage gap
  stays visible per the project's own stated practice.
- `ml.data_pipeline.tasks` -- all 7 existing task definitions, task**D** (`authorized vs
  unauthorized-OR-none`) is the primary framing per the existing README, matching this project's actual
  question ("is an authorized person here right now").
- `ml.data_pipeline.features.build_feature_matrix` -- the per-subcarrier mean/std/skew/kurtosis
  (amplitude + phase) + RSSI feature vector, used as-is for every classical (SVM/GBM) model below.
- `ml.training.train_day3_ch6_model.build_day3_ch6_manifest` -- the exact channel-6, 5-date-pooled,
  single-receiver-collapsed manifest (also already excludes `eval_day3ch6_holdout.HOLDOUT_SESSIONS`,
  which stays held out here too for comparability with that script's own before/after numbers).
- `ml.training.compare_receiver_fusion.build_3receiver_manifest` / `build_fused_window_index` /
  `FusedWindowDataset` -- the 3-receiver alignment-by-position dataset for 2026-09-21/22.
- `ml.evaluation.metrics.compute_eer` / `compute_auroc` -- shared scoring for every model, classical or
  deep, so numbers are directly comparable across the whole battery.

## What's new in `ml_2` (the actual deliverable)

1. **`models/svm_gait.py`** -- priority per this round's explicit ask ("SVM gait detection... one of
   the papers said it works", referring to Wii, Sensors 2017). Three variants on the same handcrafted
   feature vectors:
   - One-Class SVM (RBF), fit on authorized-only features -- the direct open-set reject-gate analog of
     Wii's GMM stage.
   - Binary SVC (RBF, balanced class weight) on taskD -- classical closed-set sibling to the existing
     RandomForest baseline, so SVM vs RandomForest is a direct, controlled comparison.
   - GMM (stranger-reject) + multiclass SVC (identity among anjali/barath) -- the literal two-stage Wii
     replica.
2. **`models/gbm_baseline.py`** -- XGBoost on the same features. Cheap addition; its
   `feature_importances_` doubles as the "which subcarriers/stats actually carry signal" diagnostic
   flagged in `OPEN_SET_MODEL_STRATEGY.md`.
3. **`models/backbone.py` + `models/prototypical.py`** -- the piece the existing model zoo explicitly
   lacks. `taskF_identity_or_nonauth`'s own docstring in `ml/data_pipeline/tasks.py` says as much: *"this
   is still a closed-set softmax classifier ... not expected to fully solve open-set generalization ...
   see the embedding/centroid approach for that."* This is that embedding/centroid approach: per-identity
   centroids in embedding space (reusing the existing `BranchEncoder` transformer block for the
   amplitude/phase branches, so it stays consistent with the rest of the model zoo), classified by
   distance, with a CAUTION-style distance-ratio intruder threshold calibrated using only held-out
   *known* identities (no real stranger data needed to set the threshold).
4. **`models/arcface.py`** -- same backbone, additive-angular-margin loss instead of prototypical loss,
   as a second embedding-based open-set candidate to compare against prototypical directly.
5. **GPU**: the existing `ml/training/train.py::train_classifier` and
   `ml/training/compare_receiver_fusion.py::train_fused_classifier` hardcode `torch.device("cpu")`
   ("CPU-only budget" was a deliberate prior constraint). `ml_2/training/gpu_utils.py` provides a
   drop-in equivalent that auto-selects CUDA, used for every deep model here (new and re-run existing).
   Confirmed working: 2x NVIDIA L4 visible, `torch` reinstalled from the plain PyPI wheel (was a
   CPU-only build) to pick up CUDA 13 support.
6. **`training/train_transformer_gpu.py`** -- re-runs the existing `WhoFiTransformer`,
   `DualBranchTransformer`, `CrossAttentionTransformer` on the channel-6-pooled dataset, on GPU, under
   the open-set and cross-day splits (not just the single 80/20 split `run_stage2.py` used under the
   CPU budget). `transformer_calib_context` and `cascade` are deliberately **not** included in this
   first battery -- see the open questions at the end of this doc.
7. **`training/train_receiver_fusion_gpu.py`** -- re-runs the existing `ReceiverFusionTransformer` (the
   already-implemented 3-ESP fusion model) on GPU, same open-set/cross-day splits, scoped to 09-21/22
   only (the only dates with 3 receivers), exactly as `compare_receiver_fusion.py` already scopes it.
8. **`evaluation/backtest_all.py`** -- runs every model above (classical + deep) through the same
   leave-one-unauthorized-person-out and leave-one-day-out evaluation, and writes one combined
   comparison table/CSV -- "backtest all of them at once."
9. **`training/run_all.py`** -- builds the shared channel-6 dataset once, then fires every training
   script above in sequence (not in parallel -- both GPUs are free, but sharing one dataset-in-memory
   cache and writing to shared log files sequentially is simpler and safer to get right first), then
   runs the consolidated backtest.

## Deliberately out of scope for this first pass

The full creative list in `OPEN_SET_MODEL_STRATEGY.md` (GNN-over-subcarriers, radar/micro-Doppler CNN,
MAML recalibration, CUSUM adaptive detection, contrastive/masked self-supervised pretraining, generative
hard-negative synthesis, PatchTST/Conformer/Perceiver transformer variants) is **not** implemented here.
Given the explicit ask to fire everything at once, the battery above is the set of models that (a) are
well-specified enough to implement correctly in one pass and (b) either directly answer "does SVM gait
detection work here" or fill a real, named gap in the existing model zoo (open-set embeddings). See the
question list at the end of the training run for what to prioritize next.
