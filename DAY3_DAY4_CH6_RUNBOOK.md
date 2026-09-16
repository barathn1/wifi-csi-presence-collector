# Day 3 + Day 4 / Channel 6 Runbook

Train and live-test the models scoped to Day3 (2026-09-15) **channel-6** sessions pooled with ALL of
Day4 (2026-09-16, entirely channel 6 — verified per-session from the actual `channel_primary` field,
not assumed from the date). This extends `DAY3_CH6_RUNBOOK.md`'s single-day track with a second day,
now that one exists — same architecture (WhoFi transformer), same calibA self-calibration, same
"evaluate honestly, then ship" discipline. Day3's channel-6 subset still excludes
`ml.evaluation.eval_day3ch6_holdout.HOLDOUT_SESSIONS` (kept held out for that script's own numbers).

Run everything below from the repo root, with `.venv` (or `.venv-ml` on this machine) activated.

---

## 0. Mutual exclusion + preflight

Same as `LIVE_INFERENCE_RUNBOOK.md` steps 0-1: only one process can hold the ESP32 transport at a
time (stop `collector.cli_collect` / `ml.visualization.player_server` / any other `live_infer.py`
first), and `scripts/reconnect.sh` + `scripts/connect_check.sh` must both pass before continuing.

**Before live-testing, physically confirm the router is pinned to channel 6** — `--expect-channel 6`
in step 2 below will catch a drift once the stream is running, but confirming beforehand avoids
wasting a calibration wait on a mismatched band.

---

## 1. Train (already done once)

```
python -m ml.training.train_day3_day4_ch6_model
```
What it does:
- Filters `data/manifest.csv` to `2026-09-15` + `2026-09-16` sessions, confirms each one's actual
  channel from `channel_primary` (not assumed), keeps only channel-6 sessions, and excludes Day3's
  already-established `HOLDOUT_SESSIONS`. As trained: **51 sessions** — 25 authorized (anjali/barath,
  12 from Day3 + 13 from Day4), 16 unauthorized (6 distinct strangers: divya/harshitha/sumanth/abdul
  from Day3, manas/kishore added on Day4), 10 empty-room (6 + 4).
- Re-derives the time-normalization target rate from THIS pooled dataset (native rates ranged
  130.8-270.3 Hz across all 51 sessions → target 128.20 Hz, ~1.56s/window) — the same fix
  `train_day3_ch6_model.py` applied within Day3 alone, now automatically also covering any Day3-vs-Day4
  packet-rate difference, with no extra code (`compute_target_rate_hz` always re-derives from whatever
  manifest is passed in).
- Builds one pooled `calibA` empty-room baseline from all 10 channel-6 `none` sessions across both days.
- Reports real held-out numbers FIRST (leave-one-unauthorized-person-out for taskD, now genuinely
  open-set across **6** stranger identities; full 5-fold session-disjoint for task0_presence/taskE),
  then retrains each task on 100% of the pooled data for the deployable checkpoint.

Checkpoints save as `ml/checkpoints/whofi_{task}_calibA_day3day4ch6.pt` — coexists with both the
Day1+2-pooled and Day3-ch6-only checkpoint sets; `--checkpoint-suffix _day3day4ch6` in step 2 picks
these specifically. All per-fold and final numbers are logged to
`ml/evaluation/results/day3day4_ch6_model_log.csv`.

### Real held-out numbers (2026-09-16 run, 16378 windows for taskD/task0, 10226 for taskE)

**taskD_auth_vs_nonauth — leave-one-unauthorized-person-out (6 folds, one per held-out stranger):**

| held-out stranger | accuracy | AUROC | EER | false-accept (as that stranger) | false-accept (as empty room) |
|---|---|---|---|---|---|
| abdul | 0.868 | 0.941 | 0.089 | 0.870 | 0.004 |
| divya | 0.859 | 0.935 | 0.143 | 0.536 | 0.005 |
| harshitha | 0.892 | 0.972 | 0.072 | 0.621 | 0.003 |
| kishore | 0.877 | 0.959 | 0.089 | 0.933 | 0.007 |
| manas | 0.851 | 0.963 | 0.077 | 0.571 | 0.003 |
| sumanth | 0.855 | 0.948 | 0.129 | 0.578 | 0.003 |
| **MEAN** | — | **0.953** | — | **0.685** | **0.004** |

**Read this honestly**: AUROC (0.953 mean, ranking ability) looks strong, but the fixed-0.5-threshold
false-accept-as-a-genuinely-novel-stranger rate is **68.5% on average** — worse than
`DAY3_CH6_RUNBOOK.md`'s already-weak single-day number (which ranged 39-58% across its 3 iterations).
**This model currently rejects a never-seen stranger correctly well under half the time.** It is
excellent at rejecting an *empty room* (0.4% false-accept) — that part of the job is solid — but not
at rejecting a *person who isn't enrolled*. With only 2 enrolled identities against 6 stranger
identities, this is consistent with `DAY3_CH6_RUNBOOK.md`'s own stated data-scale limitation, not a
new bug — pooling a second day added more stranger diversity to test against, which is exactly why
the honest number looks worse here than the single-day version, not better.

**task0_presence — session-disjoint 5-fold:**

| fold | accuracy | AUROC |
|---|---|---|
| 0 | 0.993 | n/a (single-class test fold) |
| 1 | 0.993 | 1.000 |
| 2 | 0.996 | 1.000 |
| 3 | 0.998 | 1.000 |
| 4 | 0.997 | 1.000 |
| **MEAN** | **0.996** (range 0.993-0.998) | **1.000** |

Presence detection remains the strongest, most reliable signal — as it has been throughout this
project.

**taskE_motion_standing_vs_walking — session-disjoint 5-fold:**

| fold | accuracy | AUROC |
|---|---|---|
| 0 | 0.812 | 0.908 |
| 1 | 0.753 | 0.774 |
| 2 | 0.820 | 0.881 |
| 3 | 0.854 | 0.914 |
| 4 | 0.839 | 0.905 |
| **MEAN** | **0.815** (range 0.753-0.854) | **0.877** |

Motion is mid-pack here — noticeably better than some earlier same-day runs in this project, but
still the least consistent of the three (widest fold-to-fold range).

**Final checkpoints' in-sample sanity accuracy** (NOT a generalization estimate — trained and scored
on the same 100% pooled data, only a "did training work at all" check): taskD 0.915 (trivial baseline
0.538), task0_presence 0.999 (trivial baseline 0.624), taskE_motion 0.919 (trivial baseline 0.604).
All comfortably above their trivial baselines, confirming training converged — use the held-out tables
above for the real reliability read, not these.

---

## 2. Start live inference

```
python -m ml.inference.live_infer --checkpoint-suffix _day3day4ch6 --expect-channel 6 --aggregate-windows 60 --calib-seconds 60
```
- `--checkpoint-suffix _day3day4ch6` loads the checkpoints from step 1.
- `--expect-channel 6` hard-gates on the exact channel (channel 6 and channel 11 are both 20MHz, so
  bandwidth alone can't distinguish them).
- `--calib-seconds 60` gives a longer, more stable calibration baseline (same reasoning as the other
  runbooks). `--aggregate-windows` defaults to 60; each window here is ~1.56s (this dataset's derived
  rate, wider than the Day3-only track's window), so 60 windows ≈ **~94s** of aggregation — noticeably
  longer than the ~30-60s windows in the other two runbooks. Adjust down (e.g. `--aggregate-windows 20`
  for ~30s) if you want faster-updating readings at the cost of more flicker.

---

## 3. Test protocol — exact timings, what to do, what to expect

**Calibration (0:00 for 60s)**: the moment you start step 2, it prints
`=== CALIBRATION: stand OUTSIDE the room / away from the sensor now ===`. **Nobody enters or moves
near the sensor for the full 60 seconds** — this builds a fresh empty-room baseline for your room/day
right now. Wait for `calibration complete: NNNN packets, amp_mean range [...], amp_std range [...]`
with sane (non-zero, non-NaN) numbers.

**Phase 1 — confirm empty reads empty**: still nobody in the room. Expect
`Presence: EMPTY | Auth: n/a (no one present) | Motion: n/a (no one present)`.

**Phase 2 — authorized person, standing**: anjali or barath walks in and stands still. Given each
aggregate window here is ~1.56s, give it a good ~90s+ before judging the aggregate column (the
`avg` reading), not the flickering instantaneous one. Expect `Presence: OCCUPIED` quickly, `Motion:
STANDING` to firm up. **Watch `Auth` closely and don't over-trust a single AUTHORIZED reading** — per
the held-out numbers above, this checkpoint set is good at telling occupied-from-empty but currently
weak at telling an enrolled person from an unenrolled one under repeated testing.

**Phase 3 — same person, walking**: walk around continuously for the same ~90s+. Expect `Motion` to
transition to `WALKING`; `Auth` should stay `AUTHORIZED` (or continue to be unreliable, consistent
with step 1's numbers — that's the point of testing it).

**Phase 4 (the important one) — a genuine stranger**: repeat Phase 2/3 with someone who is **NOT**
anjali/barath. **Given the 68.5% mean false-accept rate measured above, expect this to say
`AUTHORIZED` more often than not** — that is the expected, already-measured behavior for this
checkpoint set, not a surprise. Use someone NOT already in training (not divya/harshitha/sumanth/
abdul/manas/kishore) if you want a genuinely novel open-set test comparable to step 1's methodology.

**Stop** with Ctrl+C.

---

## 4. Honest reliability

See the tables in step 1 — those are the trustworthy numbers for this checkpoint set, not the final
checkpoints' in-sample accuracy. Summary:
- **Presence**: strong and consistent (99.6% mean accuracy, AUROC 1.000). Trust this signal.
- **Auth**: strong ranking ability (AUROC 0.953) but a **weak, unreliable stranger-rejection rate at
  the fixed 0.5 threshold (68.5% mean false-accept)** — currently the least trustworthy of the three
  for its actual job (telling an enrolled person from an intruder). Rejecting an *empty room* as
  "not authorized" works well (0.4% false-accept); rejecting a *real stranger* does not.
- **Motion**: mid-pack (81.5% mean accuracy, AUROC 0.877), the most fold-to-fold variable of the three.
- **Small enrolled-identity pool**: only 2 authorized identities (anjali, barath) against 6 stranger
  identities. This is very likely the dominant cause of the weak Auth number — not a code bug, and
  not something this pooling step was expected to fix (pooling adds more data and more stranger
  diversity to evaluate against, which is why the honest number here is harder to look good on than
  the single-day track's).
- **Two days, one room**: broader than the single-day Day3-ch6 track, still narrower than the
  Day1+2-pooled model (validated across two separate rooms/setups over a longer span). Says nothing
  about a Day 5.

---

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `channel MISMATCH: expected channel 6, got channel 11` | Router drifted off channel 6. Fix the router, don't `--force` past this. |
| `missing checkpoint ... _day3day4ch6.pt` | Run step 1 first. |
| Anything else (bandwidth mismatch, connection errors, slow calibration) | Same causes/fixes as `LIVE_INFERENCE_RUNBOOK.md` section 6. |

---

## 6. What this is NOT

- **Not a fix for the stranger-rejection weakness.** That's a data-scale limitation (2 enrolled
  identities), not something this pooling step was meant to or did solve — see step 4's honest table.
- **Not a replacement for the other two checkpoint sets.** All three (Day1+2-pooled, Day3-ch6-only,
  Day3+4-ch6-pooled) coexist; pick whichever matches what you're trying to learn.
- **Not a Day-5 collection/labeling pipeline.** If you collect more data, re-run step 1 to fold it in.
