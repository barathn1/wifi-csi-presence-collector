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
  the clock time), keeping only channel-6 sessions. As of this writing that's 26 sessions: 12 authorized
  (anjali/barath), 8 unauthorized (4 distinct strangers — divya, harshitha, sumanth, abdul — each
  standing+walking), 6 empty-room.
- Builds a `calibA` empty-room baseline from just those 6 channel-6 `none` sessions (not the day's
  channel-11 `none` sessions too — that would silently mix bands into the baseline).
- Reports **real held-out numbers before shipping anything**:
  - `taskD_auth_vs_nonauth`: leave-one-unauthorized-person-out — 4 folds, each holding out one of the 4
    stranger identities entirely (trained on the other 3 + all authorized/none, tested against the
    held-out one). This is a genuine open-set check: does the model reject someone it has never seen,
    not just the specific strangers it trained on.
  - `task0_presence` / `taskE_motion_standing_vs_walking`: a single session-disjoint 80/20 split.
- Only after printing those does it retrain each task on 100% of the ch6 Day3 data for the deployable
  checkpoint (same "no held-out split, this IS the shipped artifact" philosophy as `train_final_model.py`
  — the held-out numbers above are what stand in for a generalization estimate this time).

Checkpoints save as `ml/checkpoints/whofi_{task}_calibA_day3ch6.pt` — a different filename than the
Day1+2-pooled ones, so both sets coexist; this never overwrites `train_final_model.py`'s output.

All per-fold and final numbers are logged to `ml/evaluation/results/day3_ch6_model_log.csv`
(`stage=eval_loo`/`eval_split` rows are the trustworthy ones; `stage=final` is in-sample-only, same
caveat as the Day1+2 model).

**Headline held-out results (real numbers, from the 2026-09-15 run — `ml/evaluation/results/day3_ch6_model_log.csv`):**

| Task | Metric | Result |
|---|---|---|
| `taskD_auth_vs_nonauth`, mean over 4 held-out strangers | AUROC / false-accept rate | **AUROC 0.764 / false-accept 39.0%** |
| `taskD_auth_vs_nonauth`, per held-out stranger | AUROC / false-accept | abdul 0.709/55.9%, divya 0.675/48.3%, harshitha 0.814/29.1%, sumanth 0.857/22.6% |
| `task0_presence` (80/20 split) | accuracy / AUROC | 99.7% / 1.000 |
| `taskE_motion_standing_vs_walking` (80/20 split) | accuracy / AUROC | 90.1% / undefined (see note) |

**Read this honestly, don't just skim the headline AUROC:**
- **taskD is the one that matters most (it's the actual security question) and it is NOT good yet.** A
  39% average false-accept rate against a genuinely unseen stranger means well over 1 in 3 windows
  wrongly says AUTHORIZED for someone who isn't. It's also wildly inconsistent per person (22.6% for
  sumanth vs 55.9% for abdul) — with only 4 stranger identities and 2 authorized identities, this is a
  small, high-variance sample, not a settled number. This is a per-~1s-window number, not the
  30-second-aggregate protocol that got the Day1+2 model down to 5.5% — aggregating over the live session
  (step 2's `--aggregate-windows 120`) should help, but hasn't been separately validated at this
  aggregation length for this checkpoint. Don't trust a live `Auth: AUTHORIZED` reading strongly yet;
  treat this checkpoint as a work-in-progress, not a validated auth gate.
- **task0_presence is excellent (AUROC 1.000)** — occupied-vs-empty is a much easier task (large physical
  effect on amplitude/RSSI), and this is consistent with that pattern showing up elsewhere in this
  project. Trustworthy for "is anyone in the room," within the limits of a session-disjoint (not
  room-disjoint) test — it's only ever seen this one room.
- **taskE's AUROC is `nan`, not a bug**: with only ~20 sessions split by session-disjoint grouping (not
  stratified by motion label), this particular 80/20 fold happened to land only one motion class in the
  test set, which makes AUROC mathematically undefined (`compute_eer`/`compute_auroc` return `nan`
  explicitly in that case rather than a misleading number). The 90.1% accuracy is real but should be
  treated as a rough read, not a tight estimate — re-running with a different seed/fold would give a
  cleaner check.

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
