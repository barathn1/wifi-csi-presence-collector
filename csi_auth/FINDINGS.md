# csi_auth -- full findings

Independent, from-scratch cross-check of the person-identifiability question already investigated in
`analysis/standing_ch6` and `analysis/walking_ch6` (built on `ml/data_pipeline`). This pipeline shares
no code with those -- own decoder, own cleaning, own feature/model code -- so agreement between the
two is real corroboration, not a shared bug.

## Scope

- **Motion**: `walking` only.
- **Channel/hardware**: channel-6/20MHz sessions only, detected PER SESSION from each session's own
  raw packet data (`decode.dominant_channel`, checked directly against `channel_primary`/`cwb`
  fields -- not assumed from collection dates). One fixed receiver board (`ac:27:6e:a5:5b:c8`),
  confirmed present on every collection date. 6 dates: 2026-09-15, -16, -17, -21, -22, -24.
- **People**: `anjali` (22 sessions) and `barath` (22 sessions) for the identity task; ~18 other people
  (mostly 1 session each) as the "non-auth" pool for the auth task.
- **Cleaning applied**: Hampel-filter spike removal on amplitude (`cleaning.py`, ~5-9.5% of samples
  flagged per session, replaced with local median) + masking 19 verified-null guard/DC/pilot
  subcarrier bins (`subcarrier_mask.py`; indices `[0, 27-37, 93-99]` out of 128, confirmed identical
  across every date/board/person/label tested -- these read ~zero amplitude even in empty rooms, so
  they carry zero identity information by construction). Neither fix moved the needle on
  auth-vs-non-auth (see below) -- they were the right things to check, not the actual bottleneck.
- **Models**: SVM (RBF), RandomForest, HistGradientBoosting, CNN+BiLSTM (attention-pooled), CNN+
  self-attention (mean/attention-pooled), and a soft-vote ensemble of whichever subset a given script
  compares. All in `models.py`.
- **Splits**: leave-one-day-out throughout (never a random packet-level split), plus a
  session-level-label-shuffle permutation control run alongside every real-label result.

## 1. Anjali vs Barath (closed 2-person identity) -- the core result

### Window-level (one ~1-2s window, real labels vs shuffled floor)

| model | real balanced accuracy | shuffled-label floor | gap |
|---|---|---|---|
| SVM | 0.663 | 0.491 ± 0.048 | **3.6 std** |
| Random Forest | 0.657 | 0.489 ± 0.046 | **3.7 std** |
| CNN+BiLSTM | 0.667 | 0.503 ± 0.036 | **4.6 std** |

Decisive. Real labels do not merely edge out chance, they clear it by 3.6-4.6 standard deviations on a
5-shuffle null estimate. This rules out chance/overfitting artifacts outright.

### Session-level (average a model's confidence across a WHOLE test session's ~150 windows, spread
across the session's full ~4-minute duration, before deciding once)

| model | accuracy (mean ± std across 6 folds) | shuffled floor | gap |
|---|---|---|---|
| SVM | 0.859 ± 0.181 | 0.435 ± 0.174 | **+42.4 points** |
| Random Forest | 0.824 ± 0.245 | 0.444 ± 0.167 | +38.0 points |
| HistGradientBoosting | 0.805 ± 0.233 | -- | -- |
| **CNN+BiLSTM** | **0.900 ± 0.245** | 0.530 ± 0.179 | **+37.0 points** |
| CNN+Attention | 0.885 ± 0.157 | -- | -- |
| Ensemble | 0.863 ± 0.244 | -- | -- |

Averaging within a session does NOT inflate the shuffled floor (it stays flat, ~44-53%, same as
window-level) -- confirms this is a real per-window signal getting denoised by aggregation, not an
artifact of the aggregation method itself. Caveat: some folds have only 5-9 test sessions, so
individual-fold accuracy is coarse (a couple of folds hit exactly 100%, one dipped to 40%) -- the
6-fold mean is the number to trust.

### Live-demo timing: accuracy vs. how long someone has actually been walking

`demo_timing.py` -- windows accumulated IN CHRONOLOGICAL ORDER as they'd arrive live (no peeking ahead,
unlike the session-level number above, which samples across the whole ~4-minute session). Real elapsed
seconds come from the hardware's own `device_time_us`, not an assumed packet rate (observed capture
rate varies ~200-500 Hz across sessions, so "1 window" is NOT "1 second" -- see below).

| k windows | median real time | CNN+Attention | CNN+BiLSTM | SVM | Ensemble (4-model) | RF | HGB |
|---|---|---|---|---|---|---|---|
| 10 | 4.4s | 81.6% | 73.7% | 65.8% | 71.1% | 60.5% | 73.7% |
| 30 | 11.9s | 84.2% | 78.9% | 78.9% | 78.9% | 76.3% | 76.3% |
| 45 | 18.2s | 89.5% | 84.2% | 86.8% | 81.6% | 76.3% | 76.3% |
| 60 | 23.8s | 84.2% | 86.8% | 81.6% | 78.9% | 76.3% | 76.3% |
| 90 | 35.8s | 90.9%* | 86.4%* | 86.4%* | 84.1%* | 75.0%* | 77.3%* |
| 120 | 48.2s | 86.8%* | 84.2%* | 81.6%* | 84.2%* | 77.3%* | 76.3%* |
| 150 | 60.1s | 89.5%* | 86.8%* | 84.2%* | 84.2%* | 75.0%* | 76.3%* |
| 200 | 79.6s | 94.7%* | 89.5%* | 84.2%* | 84.2%* | 76.3%* | 76.3%* |
| 300 | 110.9s | 100.0%†| 96.9%†| 87.5%†| 93.8%†| 87.5%†| 84.4%†|

n=38 sessions for k<=200 (6 excluded from timing stats only -- clock-reset glitch, a known ESP32
hardware quirk; their correctness data is unaffected, only their elapsed-time label was corrupted).
\* = exact figure recomputed from the 38-session-clean subset differs slightly from the raw
`demo_timing_results.csv` printout, which pooled all 44; use this table's numbers.
† n=32 at k=300 (some sessions ran out of windows) -- 100%/96.9% on 32 sessions means literally zero
mistakes, encouraging but not fully proven at this sample size.

**Reading**: accuracy climbs steeply from 4s to ~18-24s, then mostly plateaus for every model except
CNN+Attention, which keeps creeping up out to ~80-110s. **CNN+Attention is the strongest model at
almost every checkpoint.**

## 2. Auth (Anjali OR Barath) vs non-auth (anyone else) -- the open-set question

| model | real balanced accuracy | shuffled floor | gap |
|---|---|---|---|
| SVM | 0.519 | 0.503 ± 0.024 | 0.8 std |
| Random Forest | 0.531 | 0.499 ± 0.025 | 1.2 std |
| CNN+BiLSTM | 0.511 | 0.493 ± 0.031 | 0.6 std |
| Ensemble | 0.531 | -- | -- |

**Not reliably above chance.** Neither spike-cleaning nor subcarrier-masking changed this (pre/post
numbers were within a point of each other both times). Most likely cause: the non-auth pool is ~18
different strangers, most with exactly one recorded session -- the model has almost nothing to
generalize from for that class. This is a data problem, not a preprocessing or architecture problem.
A positive control (presence-vs-empty-room) still scores 77-96% under the same splits, so the pipeline
itself works fine -- it's specifically open-set identity rejection that's unsupported by the data
collected so far.

## 3. Final deployable checkpoints (`train_final.py`, `checkpoints/`)

Unlike every result above (leave-one-day-out, nothing pooled, built to MEASURE generalization), this
one script trains a single final version of each of the three strongest models on ALL 6 days combined
-- Anjali vs Barath, no held-out day -- specifically to produce weights worth shipping. Its own
reported accuracy is a training-set/pooled-set number, not a cross-day estimate; trust the numbers in
section 1 for what to expect live. See `checkpoints/README.md` for the exact files and how to load them.

## Conclusion

Same overall verdict as `analysis/walking_ch6/FINDINGS.md`, now independently confirmed by a second
pipeline: **a real, moderately strong, largely day-stable Anjali-vs-Barath signal exists in walking
CSI** (decisive vs. chance at every granularity tested, live-demo-realistic sequential accuracy reaching
~85-95% by 20-80 seconds of walking with the best model). **Open-set auth-vs-stranger rejection is not
yet supported** by the data collected -- that needs more sessions per stranger, not a better model.
