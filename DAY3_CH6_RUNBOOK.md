# Day 3 / Channel 6 Runbook

Train and live-test the models that are scoped to **only** the Day3 (2026-09-15) sessions recorded on
**channel 6** — not pooled with Day1/Day2 (captured on different, non-overlapping RF bands, see
[[project-day2-cross-channel-root-cause]]) and not even pooled with this same day's channel-11 sessions
(the router was switched mid-day; same reason). See `LIVE_INFERENCE_RUNBOOK.md` for the original
Day1+2-pooled model — that checkpoint set hit a real generalization gap against live Day3 data (empty
rooms read OCCUPIED, a real authorized person never crossed the AUTHORIZED threshold; see
[[project-ml-pipeline-status]]'s "First real Day3 test" section), which is why this separate, smaller,
single-band track exists.

Run everything below from the repo root, with `.venv` activated.

---

## 0. Mutual exclusion + preflight

Same as `LIVE_INFERENCE_RUNBOOK.md` steps 0-1: only one process can hold the ESP32 transport at a time
(stop `collector.cli_collect` / `ml.visualization.player_server` first), and
`scripts/reconnect.sh` + `scripts/connect_check.sh` must both pass before continuing.

**Before live-testing, physically confirm the router is pinned to channel 6** — `scripts/check_bandwidth_live.sh`
reports bandwidth but not the exact channel; `--expect-channel 6` in step 2 below will catch a drift, but
only once the stream is already running.

---

## 1. Train

```
python3 -m ml.training.train_day3_ch6_model
```

What it does:
- Filters `data/manifest.csv` to `2026-09-15` sessions, then confirms each one's actual channel from the
  recorded per-packet `channel_primary` field (`decode_csi.py::channel_width_summary` — not assumed from
  the clock time), keeping only channel-6 sessions and explicitly excluding
  `ml.evaluation.eval_day3ch6_holdout.HOLDOUT_SESSIONS` (sessions recorded after training, kept genuinely
  held out). As of this writing that's 26 training sessions: 12 authorized (anjali/barath), 8
  unauthorized (4 distinct strangers — divya, harshitha, sumanth, abdul — each standing+walking), 6
  empty-room.
- Derives a common time-normalization rate from THIS dataset's own sessions (never a hardcoded constant —
  see `ml/data_pipeline/time_resample.py`) and resamples every session's packet stream onto it via
  bin-averaging (every packet contributes, none discarded) before windowing. This closes a real,
  confirmed confound (see [[project-day3-ch6-packet-rate-confound]] and the "what changed" section below)
  where training-authorized sessions happened to be captured at a different packet rate than
  training-unauthorized ones, which fixed-packet-count windowing turned into a non-biometric shortcut.
- Builds a `calibA` empty-room baseline from just those 6 channel-6 `none` sessions (not the day's
  channel-11 `none` sessions too — that would silently mix bands into the baseline), computed over the
  same time-normalized representation the windows use.
- Reports **real held-out numbers before shipping anything**:
  - `taskD_auth_vs_nonauth`: leave-one-unauthorized-person-out, including `none` sessions in every fold —
    4 folds, each holding out one of the 4 stranger identities entirely (trained on the other 3 + all
    authorized/none, tested against the held-out one + a held-out none slice). Genuine open-set check:
    does the model reject someone it has never seen, not just the specific strangers it trained on.
  - `task0_presence` / `taskE_motion_standing_vs_walking`: ALL 5 session-disjoint folds (not just one
    arbitrary fold), mean + range reported.
- Only after printing those does it retrain each task on 100% of the ch6 Day3 data for the deployable
  checkpoint (same "no held-out split, this IS the shipped artifact" philosophy as `train_final_model.py`
  — the held-out numbers above are what stand in for a generalization estimate this time).

Checkpoints save as `ml/checkpoints/whofi_{task}_calibA_day3ch6.pt` — a different filename than the
Day1+2-pooled ones, so both sets coexist; this never overwrites `train_final_model.py`'s output.

All per-fold and final numbers are logged to `ml/evaluation/results/day3_ch6_model_log.csv`
(`stage=eval_loo`/`eval_split` rows are the trustworthy ones; `stage=final` is in-sample-only, same
caveat as the Day1+2 model).

### What changed and why (2026-09-16 audit + fix)

A live-test session revealed real authorized-person accuracy was much worse than the pre-deployment
numbers suggested. A 5-dimension audit (parallel review + adversarial verification) found the root cause
plus 4 smaller bugs, all now fixed — see [[project-day3-ch6-packet-rate-confound]] for the full
methodology. Headline: **training-authorized sessions were captured at 146-222Hz, training-unauthorized
at 226-268Hz** (real network conditions at the time of day each block was recorded — not channel or
bandwidth, both fixed at channel 6/20MHz throughout). Fixed-packet-count windowing turned that into a
shortcut: window *duration* secretly tracked the auth label. New post-training holdout sessions (2 more
barath, 2 from a new stranger "siva") both landed in the "unauthorized-shaped" fast-rate band, which is
why a genuinely authorized person scored badly.

**Real before/after, measured against the 8 genuinely-held-out post-training sessions**
(`ml/evaluation/eval_day3ch6_holdout.py`), not just training-data folds:

| | Before any fix | After bug fixes only | After time-normalization fix |
|---|---|---|---|
| New authorized session #1 (barath), aggregate accuracy | 74-86% @60s, inconsistent | *(bugs don't touch this)* | **100% by ~23s** |
| New authorized session #2 (barath), aggregate accuracy | 61-98% @60s, inconsistent | *(bugs don't touch this)* | **100% by ~23s** |
| New stranger "siva" (walking), correctly rejected? | No — false-accepted, got worse with aggregation (→0%) | — | **Still no — worse (0%).** Not fixed by this change; likely a real small-sample gait-similarity limit (only 2 authorized identities to learn "authorized" from), not a code bug. |
| New stranger "siva" (standing), correctly rejected? | Mostly yes (95-100% @90s) | — | Yes, improved (100% by ~70s) |
| Presence, all 8 holdout sessions | ~90-100%, converges ~30-60s | — | Still ~99-100% throughout — never the problem |

**Current best checkpoints**: the 3 files in `ml/checkpoints/whofi_*_calibA_day3ch6.pt` (the time-normalized
version above) are the current best for this track, and `ml/inference/live_infer.py --checkpoint-suffix
_day3ch6` already points at them. A dated backup copy is preserved at
`ml/checkpoints/best_day3ch6_timenorm_20260916/` (gitignored like the rest of `ml/checkpoints/`, so purely
local) so a future experiment (e.g. the queued UniFi-style architecture change) that reruns
`train_day3_ch6_model.py` and overwrites the live filenames doesn't silently lose this validated result —
restore by copying those 3 files back over the live ones if a later change turns out worse.
**Training-data fold numbers moved the OPPOSITE direction** (taskD mean false-accept-vs-stranger: 39% →
45% → 58% mean across the 3 versions) — this is not a regression. The packet-rate shortcut was internally
consistent for that training day's own unauthorized cohort (all recorded in the same fast-rate block,
held-out stranger included), so it was propping up the same-day fold number while actively hurting
generalization to anything recorded at a different pace — exactly the real holdout sessions. Removing it
trades a fake same-day number for real accuracy on genuinely new data, which is the right trade; don't be
alarmed that the training-fold number looks worse now.

**Bottom line**: the fix solved "does a real authorized person get recognized" (the main complaint). It
did NOT solve "does every stranger get reliably rejected" — that remains weak and is a data-scale
limitation (2 authorized identities, 5 stranger identities total), not something more debugging fixes.
Next step queued: a UniFi-style (arXiv:2512.22143) time-aware attention model that learns directly from
irregular-rate sequences instead of resampling at all — not yet implemented.

---

## 2. Start live inference

```
python3 -m ml.inference.live_infer --checkpoint-suffix _day3ch6 --expect-channel 6 --aggregate-windows 120 --calib-seconds 60
```
- `--checkpoint-suffix _day3ch6` loads the checkpoints from step 1 instead of the Day1+2-pooled default.
- `--expect-channel 6` hard-gates on the exact channel, not just bandwidth — channel 6 and channel 11 are
  both 20MHz, so `--expect-mhz 20` alone can't tell them apart. If the router has drifted off channel 6,
  this aborts loudly instead of silently feeding the model data from a band it never saw.
- `--aggregate-windows 120 --calib-seconds 60` give a ~60s decision window and a longer, more stable
  calibration — same reasoning as the Day1+2 runbook, not re-validated at this window size for this
  checkpoint specifically.

---

## 3. Test protocol — exact timings, what to do, what to expect

**Calibration (0:00-1:00, 60 seconds). Answer to "do I need to stay out of the room": yes — nobody
enters, nobody moves near the sensor, for this entire 60 seconds.** The moment you start the command in
step 2, it prints:
```
=== CALIBRATION: stand OUTSIDE the room / away from the sensor now ===
```
This is building a fresh empty-room baseline (`calibA`) for the room/day you're testing in right now —
walking in during this window contaminates the baseline with your own reflections, and every downstream
reading (Presence/Auth/Motion) will be measured relative to a wrong "empty" reference for the rest of the
session. Wait for:
```
calibration complete: NNNN packets, amp_mean range [...], amp_std range [...]
```
with sane (non-zero, non-NaN) numbers before anyone approaches. If it takes noticeably longer than 60s,
that's fine — see the troubleshooting table below (it's a safety floor waiting for enough packets, not a
bug).

**Phase 1 — confirm empty reads empty (1:00-2:00, 60 seconds)**: still nobody in the room. Expect:
```
Presence: EMPTY (0.08, 24/120 avg)  |  Auth: n/a (no one present)  |  Motion: n/a (no one present)
```
Auth/Motion intentionally go quiet while Presence says nobody's there.

**Phase 2 — authorized person, standing (2:00-3:00, 60 seconds)**: now the authorized person (anjali or
barath) walks in and stands still. 60 seconds matches `--aggregate-windows 120` (~2 windows/second), so
give it the full 60 seconds before judging. Expect `Presence: OCCUPIED` almost immediately, and
`Auth: AUTHORIZED` / `Motion: STANDING` to firm up in the aggregate (`avg`) column by the end. Trust the
aggregate column, not the flickering instantaneous one before it.

**Phase 3 — same person, walking (3:00-4:00, 60 seconds)**: walk around continuously (don't stop and
start) for the full 60 seconds. Expect `Motion` to transition to `WALKING` as the rolling window catches
up; `Auth` should stay `AUTHORIZED` throughout.

**Phase 4 (recommended) — an unauthorized person (4:00-6:00, 2x60 seconds: standing then walking)**:
repeat Phase 2 then Phase 3 with someone who is NOT anjali/barath. Expect `Presence: OCCUPIED` and a
`Motion` reading as before, but `Auth: NOT AUTHORIZED`. **Use someone who is NOT one of the 4 strangers
already in the training data (divya, harshitha, sumanth, abdul)** if you want this to be a genuine
live open-set check comparable to the leave-one-out evaluation in step 1 — re-testing with one of those 4
only confirms the model remembers a known identity, not that it generalizes to a new one.

**Stop** with Ctrl+C.

---

## 4. Honest reliability

See the headline results table in step 1 — those leave-one-stranger-out/held-out-split numbers are the
trustworthy estimate for this checkpoint set. Caveats specific to this track, on top of the usual
fixed-0.5-threshold caveat (same as the Day1+2 model, see `LIVE_INFERENCE_RUNBOOK.md` section 5):
- **Small dataset**: a few dozen sessions and 4 stranger identities is enough for a first real read, not
  a statistically tight one. Treat any single number as directional; re-run after collecting more ch6
  Day3 data (more sessions and/or more stranger identities) before trusting it the way the original
  Day1+2 sweep's numbers were trusted.
- **Single room/day**: unlike the Day1+2 model (validated across two separate collection days), this
  model has only ever seen one room on one day. It says nothing about whether it'd hold up on a Day 4.

---

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `channel MISMATCH: expected channel 6, got channel 11` | Router drifted (or was never switched) off channel 6. Fix the router, don't `--force` past this — the whole point of this checkpoint set is being scoped to one band. |
| `missing checkpoint ... _day3ch6.pt` | Run step 1 first. |
| Anything else (bandwidth mismatch, connection errors, slow calibration) | Same causes/fixes as `LIVE_INFERENCE_RUNBOOK.md` section 6 — the underlying mechanics are shared, only the checkpoint set and channel differ. |

---

## 6. What this is NOT

- **Not a replacement for the Day1+2-pooled model or `LIVE_INFERENCE_RUNBOOK.md`.** Both checkpoint sets
  coexist; use whichever matches what you're trying to learn (broad two-day-validated behavior vs. a
  clean, single-band, single-day baseline that isn't confounded by the cross-channel issue).
- **Not a Day-4 collection/labeling pipeline.** If you collect more data, re-run step 1 to fold it in —
  this doc/checkpoint set will need updating, not automatically kept in sync.
