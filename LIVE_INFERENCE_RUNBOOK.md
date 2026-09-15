# Live Inference Runbook

Train the models once, then run them live against the ESP32's real-time CSI stream and watch them
classify presence / authorization / motion as it happens — a smoke test you can run **before** any
real, labeled "Day 3" dataset exists. See `ml/reports/day2_next_steps.md` for where the underlying
numbers (5.5% false-accept, etc.) came from.

> **⚠️ These checkpoints assume live/Day-3 data is captured at 20MHz** (matching Day 2's format, channel
> width `cwb=0` — see `ml/reports/day2_next_steps.md`'s cross-channel findings for why this matters:
> Day 1's 40MHz data and Day 2's 20MHz data sit on non-overlapping RF frequency bands, so a model
> trained assuming one doesn't transfer reliably to the other). If your router/board actually negotiates
> 40MHz for a live session, these models were never validated against that format and predictions would
> be unreliable — `live_infer.py` enforces this: it checks the live bandwidth before calibration even
> starts and **hard-aborts by default** on a mismatch (`--expect-mhz 20` is the default; pass
> `--expect-mhz 40 --force` only if you deliberately want to test against 40MHz data anyway, understanding
> the model wasn't validated for it). If you're not sure what your setup will negotiate, run
> `scripts/check_bandwidth.sh` on a fresh recording first, or pin the router to 20MHz as recommended
> earlier this project.
>
> **Bandwidth alone isn't the whole story**: channel 6 and channel 11 are both 20MHz but are different,
> non-overlapping RF bands (same disease as the Day1/Day2 40MHz-vs-20MHz mismatch, just a different
> symptom). `live_infer.py --expect-channel N` hard-gates on the exact channel too, not just bandwidth —
> see `DAY3_CH6_RUNBOOK.md` for the separate Day3-channel-6-only checkpoint track that needs this.

Run everything below from the repo root, with `.venv` activated.

---

## 0. Mutual exclusion — read this first

The ESP32's transport (TCP or serial) can only be held by **one process at a time**. Before running
`live_infer.py`, make sure `collector.cli_collect` and `ml.visualization.player_server` are both
stopped. If one of them is still holding the connection, `live_infer.py` will fail fast with a clear
`[Errno 98] Address already in use` (TCP) or a serial-port-busy error — not a silent hang — so it's
obvious what to fix.

---

## 1. Preflight

Same as `TESTING_RUNBOOK.md`:
```
scripts/reconnect.sh
scripts/connect_check.sh
```
Both must pass (`connect_check.sh` should print `OK: board ... connected: ...`) before continuing.

---

## 2. Train the models once

```
python3 -m ml.training.train_final_model
```
Trains all 3 checkpoints (presence, auth, motion) on **all** of Day1+Day2 pooled data — takes roughly
20-25 minutes total on this hardware. You'll see, for each:
```
=== taskD_auth_vs_nonauth: 30504 windows, classes=[0, 1], trivial-always-predict-majority baseline=0.7153 ===
  in-sample sanity accuracy (NOT a generalization estimate): 0.9XXX  (trivial baseline: 0.7153)
  saved -> ml/checkpoints/whofi_taskD_calibA.pt
```
**The printed accuracy is an in-sample sanity check only** — it's trained and evaluated on the exact
same data, so it isn't a real generalization estimate. Its only job is to confirm training actually
worked (i.e. it should be meaningfully above the printed "trivial baseline" — a model that just always
predicts the majority class). The real test is the live one below, against data the model never saw.

Checkpoints land in `ml/checkpoints/` (gitignored — regenerate anytime by rerunning this command; it
overwrites by default). You only need to do this once, then re-run it later if you retrain (e.g. once
Day 3 data gets folded into the training pool).

---

> **Looking for the Day-3, channel-6-only checkpoints instead?** That's a separate, smaller, single-band
> track — see `DAY3_CH6_RUNBOOK.md` for its own training command, live-inference invocation, and honest
> reliability numbers. The rest of this doc covers only the Day1+2-pooled model below.

## 3. Start live inference

```
python3 -m ml.inference.live_infer --aggregate-windows 120 --calib-seconds 60
```
`--aggregate-windows 120` gives a ~60-second decision window instead of the default ~30s (60 windows).
The validated 5.5%-false-accept number was measured at 60 windows/~30s specifically — 120/~60s is a
well-motivated extension (accuracy kept improving as the window got longer, up through 60), not a
re-confirmed result. `--calib-seconds 60` extends the live empty-room calibration from the default 15s
to 60s — a longer calibration period gives a more stable baseline estimate (more packets averaged into
the mean/std), at the cost of a longer wait before testing starts. Drop either flag to go back to the
defaults (~30s aggregate window, 15s calibration).

---

## 4. The test protocol — exact timings, what to do, what to expect

**Calibration (0:00–1:00, ~60 seconds)**: the script prints
`=== CALIBRATION: stand OUTSIDE the room / away from the sensor now ===`.
**Stay away from the sensor for the full 60 seconds** until it prints `calibration complete` with some
sane-looking (non-zero, non-NaN) amplitude ranges. This builds a fresh empty-room baseline for
whatever room/day you're testing in right now — the same self-calibration idea (`calibA`) that's
already proven to matter a lot in this project. Longer than the 15s default on purpose here, for a
more stable baseline estimate.

**Phase 1 — confirm empty reads empty (~1:00–2:00, 60 seconds)**: stay out of the room a little longer.
Expect:
```
Presence: EMPTY (0.08, 24/120 avg)  |  Auth: n/a (no one present)  |  Motion: n/a (no one present)
```
Auth and Motion intentionally go quiet when Presence says nobody's there — there's no point guessing
someone's motion when no one's detected.

**Phase 2 — authorized person, standing (~2:00–3:00, 60 seconds)**: the authorized person walks in and
then **stands still**. 60 seconds matches the `--aggregate-windows 120` window size (roughly 2
windows/second), so give it the full 60 seconds before judging the result. Expect `Presence: OCCUPIED`
almost immediately, and `Auth: AUTHORIZED` / `Motion: STANDING` to firm up in the aggregate column by
the end of the 60 seconds. The *instantaneous* number (before the `avg` one) will flicker — trust the
aggregate, not the instantaneous reading.

**Phase 3 — same person, walking (~3:00–4:00, 60 seconds)**: now **walk around** the room continuously
for 60 seconds (don't stop and start — keep moving the whole time). Expect `Motion` to transition to
`WALKING` as the 60-second aggregate window rolls forward past the transition point. `Auth` should stay
`AUTHORIZED` throughout — walking shouldn't change who the model thinks you are.

**Phase 4 (optional but recommended) — a different, unauthorized person (~4:00–6:00)**: repeat Phase 2
(60s standing) then Phase 3 (60s walking) with someone who is NOT one of the authorized people. This is
the one to watch most closely: expect `Presence: OCCUPIED` and a `Motion` reading as before, but
`Auth: NOT AUTHORIZED`. Per the validated numbers (measured at the ~30s window, see the caveat in step 3
above) this should be right roughly **19 times out of 20** (5.5% false-accept rate) at ~30s, and
plausibly better at ~60s — don't be alarmed by one wrong reading in a longer test session, that's within
the expected error rate, not a sign something's broken.

**Stop** with Ctrl+C.

---

## 5. Honest reliability, per signal — so you know how much to trust each one

| Signal | How good is it | Notes |
|---|---|---|
| **Auth** (30s aggregate; ~60s if using `--aggregate-windows 120`) | Strong on Day2 sweep numbers, but failed its real Day3 test | 5.5% false-accept-unauthorized, 1.6% false-accept-none at ~30s — measured on Day1/Day2 held-out folds. The first real live Day3 run (see [[project-ml-pipeline-status]]) showed a genuine generalization gap: a real authorized person's aggregate score never confidently crossed 0.5. Treat this checkpoint's live Auth output as unreliable until that gap is understood, not just "trust the aggregate." |
| **Presence** | Good ranking ability, known quirk, also failed its Day3 test | AUROC 0.823 in Day1/Day2 testing, but a documented accuracy/threshold-calibration mismatch; live Day3 testing showed the aggregate swinging 0.003-0.997 within a single empty room. |
| **Motion** (standing/walking) | Weakest of the three, experimental | AUROC ~0.64 in Day1/Day2 testing; collapsed to predicting WALKING 96-99% of the time on real Day3 data. |

All three use a fixed 0.5 decision threshold — there's no held-out data in a "train on everything" run
to fit a better one, so this is the only defensible choice, not a tuned value.

(For the Day3-channel-6-only checkpoints' reliability numbers, see `DAY3_CH6_RUNBOOK.md` section 4.)

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Address already in use` right at startup | Another process (`cli_collect.py`, `player_server.py`) is still holding the transport — stop it first (see step 0). |
| `bandwidth MISMATCH` and the script exits | The live board negotiated a different channel/bandwidth than the checkpoints assume (default 20MHz) — this is the exact Day1/Day2 failure mode this project already hit once, so it's fatal by default. Check the router hasn't drifted (`scripts/check_bandwidth.sh` on a fresh recording, or pass `--force` to proceed anyway at your own risk). |
| `missing checkpoint ...` | Run step 2 (`train_final_model.py`) first. |
| Calibration takes much longer than the nominal `--calib-seconds` to complete | Packet rate is unusually low — the script waits past the nominal duration until it has a full window's worth of packets, as a safety floor. Check `connect_check.sh` still passes and the stimulus traffic generator started (see the log line at startup). |
| `connect_check.sh` fails | Same causes/fixes as `TESTING_RUNBOOK.md`'s troubleshooting table — this script needs the exact same live connection `cli_collect.py` does. |

---

## 7. What this is NOT

- **Not a replacement for `cli_collect.py`.** Nothing gets written to `data/` or `manifest.csv` — this
  is a live smoke test of the trained models, not a data-collection tool.
- **Not a Day-3 collection/labeling pipeline.** If you want real, labeled Day-3 training data, use
  `cli_collect.py` as usual. This script is purely for watching the already-trained models work live,
  before that formal collection happens.
