# Day 2 cross-day generalization — next-steps queue

Context: the Day 2 sweep (`ml/training/run_day2_sweep.py`) was paused mid-run (Stage T1), then resumed
and left running (PID 41975 as of 2026-09-11 — check `ps aux | grep run_day2_sweep` for the current
PID, it changes across relaunches). While it ran, went through the completed logs
(`ml/evaluation/results/day2_sweep_log.csv`), researched the WiFi-CSI-sensing literature for this exact
problem (Day 1 and Day 2 were captured on non-overlapping RF frequency bands — see
`ml/data_pipeline/decode_csi.py --channel` and [[project-day2-cross-channel-root-cause]] in memory),
and implemented most of the resulting backlog. Status marked per item below.

## Status: v1, v2, v3 of the full sweep all completed (2026-09-11/12). v3 validated everything below.

## Requires the user (manual, physical access needed) — NOT DONE, needs you

1. **Pin the router's channel + bandwidth** (disable auto-channel-width, force 20MHz-only, fix a
   specific channel) before any future collection day. Every methodology source found says this is
   the field's actual answer to this problem — not a post-hoc algorithmic fix. Router admin page.
2. **Collect one "bridging" session**: same empty room, back-to-back, once under each old config (if
   still reproducible) or at minimum a fresh session immediately after pinning the channel, to get a
   supervised anchor for any future cross-config comparison. Needs the collector running physically.

## Implemented (2026-09-11)

3. **DONE.** ℓ1 per-packet gain normalization added: `calibration.py::apply_l1_gain_norm`. Not yet
   run through the full sweep (that would be a new "calibC" arm) — implemented and unit-tested only.
4. **DONE AND VALIDATED (v2/v3).** `calib_context_transformer`'s scale-mismatch fix applied in
   `ml/models/transformer_calib_context.py` (per-row standardization + learned type-embedding). Real
   before/after comparison, Stage T1 baseline config: **0.585 (v1, pre-fix) -> 0.617 (v2/v3,
   post-fix)**; 6 of 7 configs improved (heads=2 config jumped 0.319 -> 0.553). The fix works, but its
   ceiling (~0.69 at best) still doesn't beat `whofi_transformer`'s 0.749 baseline — architecture fixed,
   still not the winner.
5. **DONE AND VALIDATED (v2/v3).** `bilstm` got its own architecture ablation (`stage_t1_bilstm`,
   sweeping `hidden_size` 16/32/64 and `num_layers` 1/2/3). Result: despite being Stage B's single
   best/most-stable model overall (AUROC 0.798 there, under calibA averaged across both directions),
   under T1b's isolated raw/single-direction/capped-training test its best config only reached 0.571 —
   doesn't beat `whofi_transformer`. Not a contradiction: different preprocessing/direction being
   measured, but confirms whofi's baseline config remains the overall winner even after properly
   testing bilstm's architecture space.
6. **ANSWERED (analysis, not code).** Why `calibB` underperforms `calibA`: hypothesis is that Variant
   B's per-subcarrier `sigma_ref/sigma_today` scaling ratio is fit ONLY from empty-room (`none`)
   packets, then applied uniformly to occupied windows too — this assumes the empty-room-to-occupied
   relationship is structurally identical across days except for that one affine map. If it isn't
   (plausible given the channel-frequency mismatch — the occupied-vs-empty *difference* itself may not
   transform the same way `none`-vs-`none` does across bands), Variant B could be importing distortion
   into the very signal it's trying to preserve, on top of correcting the nuisance drift. Recommendation
   stands: keep `calibA` as the default, treat `calibB` as a secondary, not co-equal, experiment.
7. **DONE.** RSSI fusion implemented as a non-invasive wrapper (`ml/models/rssi_fusion.py`:
   `RSSIFusionClassifier` + `train_rssi_fusion_classifier`), built on `CsiWindowDataset`'s new
   `include_rssi=True` opt-in flag (default `False`, so every existing script is unaffected). Verified
   end-to-end on a small real subset (300 windows) — trains and evaluates correctly, ~73% accuracy on
   the tiny smoke sample (not a real result, just a correctness check). Not yet run as a full sweep arm.
8. **Superseded by a better fix.** Rather than just logging `|AUROC-0.5|` as a diagnostic, item 9's
   Doppler-feature result below suggests the sign-flip itself is a symptom of per-subcarrier features
   breaking across the channel mismatch — worth chasing the better representation (already done, see
   below) over just labeling the symptom. Still a good idea to add to any report-generation script later.
9. **DONE, with a real (mixed) empirical answer** — tested both options directly on real cross-day
   data, not just implemented blindly:
   - **Permutation-invariant summary stats** (`ml/data_pipeline/invariant_features.py`: percentiles,
     skew/kurtosis, spectral entropy, top-3 subcarrier-covariance eigenvalues, lag-1 autocorrelation):
     tested on a real 2400-window subsample, RandomForest, both directions, raw AND calibrated —
     **AUROC 0.48-0.51 in every configuration, i.e. no usable signal at all.** Collapsing away
     subcarrier position destroys the real discriminative signal along with the day-to-day noise; this
     approach is a dead end for this specific task on this data, worth knowing before investing further.
   - **Doppler/velocity-domain features** (`ml/data_pipeline/doppler_features.py`: Welch PSD per
     subcarrier over 3000-packet segments, averaged across subcarriers, band-power in 5 Doppler bins):
     tested on real 10-18s segments, RandomForest, calibA — **AUROC 0.605 (train Day1→test Day2) and
     0.753 (reverse direction)**. Meaningfully lower peak than the best per-subcarrier+calibA number
     (0.845) but — unlike every per-subcarrier result tested — **neither direction flips below 0.5.**
     This is the most cross-day-robust representation found so far and the most promising lead for
     further work; it just needs a longer decision horizon (10-18s, not ~1s) since gait/motion
     frequencies are slow.
10. **DONE, with a real result** — `ml/training/run_fewshot_finetune.py`: pretrains on all of Day 1,
    holds out 3 whole Day 2 sessions (one per label) as the few-shot fine-tuning pool, evaluates
    zero-shot vs. after fine-tuning on the remaining Day 2 sessions. Result:

    | | accuracy | EER | AUROC |
    |---|---|---|---|
    | zero-shot (no Day2 data) | 0.748 | 0.582 | **0.410** (below chance) |
    | few-shot, 1/3 sessions (680 windows) | 0.218 | 0.346 | 0.649 |
    | few-shot, 3/3 sessions (1139 windows) | 0.307 | 0.275 | **0.766** |

    AUROC improves monotonically and substantially with more few-shot data (0.410 -> 0.649 -> 0.766) —
    genuine evidence the FewSense-style approach works as a *ranking* fix, recovering real cross-day
    separability. But raw accuracy gets WORSE (0.748 -> 0.218), the opposite direction — classic
    decision-threshold miscalibration: fine-tuning on a small, class-imbalanced few-shot sample shifts
    the model's output distribution enough that the default 0.5 argmax threshold stops matching the
    eval set's true class balance, even though the underlying ranking (AUROC) is genuinely better.
    **Next step if pursuing this further: recalibrate the decision threshold after fine-tuning (e.g.
    pick the threshold that equalizes precision/recall on a small held-out slice, or fine-tune with
    class-balanced sampling) rather than trusting accuracy at a fixed 0.5 cutoff.** Full adversarial
    domain adaptation (CrossRF-style) was not attempted — this pragmatic version was tried first per
    the original ranking, and the AUROC result justifies revisiting it with the threshold fix.

## Ruled out

- **CSI-ratio method** (divide antenna 1 by antenna 2 to cancel phase noise): needs 2+ Rx antennas;
  the ESP32-S3 setup here is 1x1. Not applicable.

## Headline result (v3, 2026-09-12): calibA + segment aggregation, validated

Stage S's missing `calibA` arm (the gap flagged above) is fixed and validated. **This is the single
most important result of the whole project so far** for the actual auth-vs-non-auth question:

| preprocessing | n_windows | accuracy | AUROC | false-accept unauthorized | false-accept none |
|---|---|---|---|---|---|
| raw | 60 (~30s) | 0.798 | **0.899** | 58.2% | 3.9% |
| calibA | 60 (~30s) | 0.749 | 0.821 | **5.5%** | 1.6% |

Calibration and longer aggregation windows are COMPOUNDING levers, not redundant ones:
false-accept-unauthorized under calibA shrinks monotonically as the window lengthens: 21.5% (1
window) -> 13.7% (~5s) -> 13.0% (~10s) -> 8.0% (~20s) -> **5.5%** (~30s). Raw's peak AUROC is higher
(0.899 vs 0.821), but that's the wrong number to optimize for this task's actual goal — calibA is
dramatically better at the specific decision that matters (correctly rejecting a real intruder).
**Recommended production config: `whofi_transformer` + `calibA` + ~30s decision window.**

## What's next (not yet done)

- Wire ℓ1 gain norm (item 3) and RSSI fusion (item 7) into the main sweep as real arms (currently only
  unit/correctness-tested standalone, not part of the Stage B/T1/T2 comparison table).
- Given item 9's Doppler result is the most promising lead, consider making it a first-class arm of the
  main sweep (its own task/split treatment, since it needs longer segments than the base 200-packet
  window) rather than a standalone script.
- Consider whether the calibA+aggregation combination also fixes the earlier decision-threshold
  miscalibration pattern (task0_presence, few-shot fine-tuning) — false-accept-none is also down to
  1.6% here, suggesting the combination may be self-correcting the threshold issue, not just masking
  it. Worth checking explicitly with a recalibrated threshold on top, to see if there's still more
  headroom.
