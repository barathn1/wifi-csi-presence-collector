# Live test: GNN vs radar-CNN

Run the two best-performing models from `ml_2` -- `gnn_subcarrier` (SubcarrierGNN) and `radar_cnn`
(RadarInspiredCNN), the two FULL-coverage models in
`ml_2/evaluation/results/DAY_PERSON_MODEL_ACCURACY.md` -- live against the ESP32's real-time CSI
stream, side by side. Start the script, do a short empty-room calibration when prompted, then watch
both models print their own live AUTHORIZED / NOT AUTHORIZED reading + confidence, one line per line.

Both checkpoints were trained ONLY on channel-6 sessions (see `ml_2/PLAN.md`), so this hard-gates on
channel 6 / 20MHz by default and aborts if the router has drifted off it.

Run everything below from the repo root.

---

## 0. Mutual exclusion -- read this first

The ESP32's transport (TCP or serial) can only be held by **one process at a time**. Before running
`live_infer.py`, make sure `collector.cli_collect`, `ml.visualization.player_server`, and
`ml.inference.live_infer` are all stopped. If one of them is still holding the connection, this script
fails fast with a clear `[Errno 98] Address already in use` (TCP) or serial-port-busy error, not a
silent hang.

---

## 1. Preflight

Same as `TESTING_RUNBOOK.md`:
```
scripts/reconnect.sh
scripts/connect_check.sh
```
Both must pass (`connect_check.sh` should print `OK: board ... connected: ...`) before continuing.
Also physically confirm the router is pinned to **channel 6** -- `--expect-channel 6` (the default
below) will catch a drift once the stream is running, but it's faster to just check the router first.

---

## 2. Train the two checkpoints once

```
python3 -m ml_2.training.train_live_checkpoints
```
Trains `gnn_subcarrier` and `radar_cnn` on **100% of the channel-6 pooled dataset** (no held-out
split -- this is the deployable artifact, same philosophy as `ml.training.train_final_model`). For
each model you'll see:
```
=== gnn_subcarrier: 178342 windows, classes=[0, 1], trivial-always-predict-majority baseline=0.7XXX ===
  in-sample sanity accuracy (NOT a generalization estimate): 0.9XXX  (trivial baseline: 0.7XXX)
  saved -> ml_2/checkpoints/gnn_subcarrier_calibA.pt
```
**That printed accuracy is an in-sample sanity check only** -- it confirms training worked (should be
meaningfully above the trivial baseline), not a generalization estimate. The real, trustworthy
generalization numbers for these two models are already measured and written up in
`ml_2/evaluation/results/DAY_PERSON_MODEL_ACCURACY.md` and `SESSION_BREAKDOWN_SUMMARY.md` -- the live
run below is a live smoke test on top of that, not a replacement for it.

Checkpoints land in `ml_2/checkpoints/` (gitignored -- regenerate any time by rerunning this command;
it overwrites by default). You only need to do this once, then again later if you retrain.

---

## 3. Start the live test

```
python3 -m ml_2.inference.live_infer
```
Defaults: `--calib-seconds 60 --aggregate-windows 60 --expect-channel 6 --expect-mhz 20`.

Useful overrides:
```
python3 -m ml_2.inference.live_infer --calib-seconds 30 --aggregate-windows 40   # shorter calib / decision window
python3 -m ml_2.inference.live_infer --force                                     # continue past a channel/bandwidth mismatch
python3 -m ml_2.inference.live_infer --collect                                   # ALSO save this session as new training data
```

### `--collect` -- also save this session as a new labeled training session

Pass `--collect` if you want this live run to double as a real data-collection session (same
`{metadata.json, samples.npz}` output `collector.cli_collect` writes, landing under `data/` and
picked up by `data/manifest.csv` automatically). Right after preflight, you'll be prompted
interactively instead of needing CLI flags:
```
[collect] label (authorized/unauthorized/none):
[collect] person_id (required):        # skipped if label == none
[collect] motion (standing/walking):    # skipped if label == none
[collect] notes (optional):
```
Answer exactly as you would for `collector.cli_collect --label ... --person-id ... --motion ...`.

**Recording only starts once the empty-room calibration finishes**, not during it -- so the
calibration window (which is always an empty room by design) never gets saved under whatever label
you gave. If you specifically want a `none`/empty-room training session, answer `none` at the prompt
and just keep the room empty for the whole run (skipping `person_id`/`motion`, same as
`cli_collect --label none`).

Saving happens on exit (Ctrl+C), same as `cli_collect.py` -- stopping early still saves whatever was
captured so far, and `kill -9` still can't be handled (by OS design) and will lose the in-memory
session. A summary line prints on save (`wrote N samples to <dir> (M dropped, X% loss)`), and
`data/manifest.csv` is rebuilt automatically afterward.

---

## 4. What happens, in order

**Step A -- calibration (default 60s).** The moment the script connects, it prints:
```
=== CALIBRATION: stand OUTSIDE the room / away from the sensor now (60s) ===
```
**Stay away from the sensor for the full duration.** This builds a fresh empty-room baseline
(per-subcarrier mean/std) for whatever room you're testing in right now -- the same self-calibration
idea (`calibA`) already validated elsewhere in this project. Walking in during this window
contaminates the baseline, and every reading afterward is measured relative to a wrong "empty"
reference for the rest of the session. Right below the banner above, a live countdown updates in
place roughly once a second:
```
=== CALIBRATING: 42 seconds left (118 packets so far) ===
```
so you always know how much longer to stay out. Once it hits 0 and enough packets have arrived, wait
for:
```
calibration complete: NNNN packets, amp_mean range [...], amp_std range [...]
```
with sane (non-zero, non-NaN) numbers. If the countdown hits 0 but the line changes to
`calibrating... waiting for enough packets (N/200)`, that's a safety floor waiting for more packets
past the nominal duration (packet rate is unusually low), not a bug -- see troubleshooting below.

**Step B -- streaming.** Once calibration finishes, the script prints:
```
=== streaming -- have a person walk in/out as needed, Ctrl+C to stop ===
```
and then one line per window, forever, until you press Ctrl+C:
```
[14:32:07] window#   42  GNN: AUTHORIZED (0.81, 42/60 avg)  |  RADAR_CNN: NOT AUTHORIZED (0.37, 42/60 avg)
```
Each model prints independently:
- the **label** (`AUTHORIZED` / `NOT AUTHORIZED`) its rolling aggregate currently says,
- the **aggregate confidence** (0.00-1.00 -- the number you asked "which model is giving me what
  accuracy": this is that number, the model's own live positive-class probability, averaged over the
  rolling window),
- `n/target` -- how many windows have fed the rolling average so far vs. the full target
  (`--aggregate-windows`, default 60). Trust the aggregate number once `n` reaches `target`; before
  that it's still filling up and will be noisier.

**Suggested test sequence** (nothing is auto-prompted beyond calibration -- act these out yourself
while watching the output):
1. Stay out of the room ~30-60s after streaming starts -- confirm both models say `NOT AUTHORIZED`
   with low confidence (empty room).
2. Have an authorized person (anjali or barath, per the training data) walk in and stay ~60s -- watch
   whether/when each model's aggregate crosses over to `AUTHORIZED`.
3. Swap in someone NOT in the training data -- watch whether each model correctly stays at
   `NOT AUTHORIZED`. This is the harder, more informative case (open-set rejection), and per
   `DAY_PERSON_MODEL_ACCURACY.md` some strangers (e.g. divya) are known to fool both models
   consistently -- don't be surprised by that specific failure, it's a documented finding, not new.

**Stop** with Ctrl+C.

---

## 5. Reading the number honestly

- The printed confidence is each model's own live probability output, aggregated -- it is **not** a
  verified accuracy, since there's no ground-truth label during a live run. Use it to compare "how
  confident is GNN vs radar-CNN right now", not as a certified error rate.
- For the real, ground-truth accuracy of each model (measured against recorded, labeled sessions,
  broken down by day and by person), see `ml_2/evaluation/results/DAY_PERSON_MODEL_ACCURACY.md` --
  both `gnn_subcarrier` and `radar_cnn` are the two FULL-coverage columns there, meaning those numbers
  are complete and trustworthy (unlike the other, partial-coverage models in that same table).
- Both models use a fixed 0.5 decision threshold -- there's no held-out data in a
  "train on everything" checkpoint to fit a better one, so this is the only defensible default, not a
  tuned value.
- GNN and radar-CNN see the data differently (GNN: per-subcarrier amplitude+phase stats over a fixed
  frequency-proximity graph; radar-CNN: an amplitude-only spectrogram, no phase at all -- see
  `ml_2/MODEL_TECHNICAL_REFERENCE.md`), so it's normal and expected for them to disagree on some
  windows. A persistent disagreement on the same person/room is itself useful signal, not noise to
  ignore.

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Address already in use` right at startup | Another process (`cli_collect.py`, `player_server.py`, `ml.inference.live_infer`) is still holding the transport -- stop it first (see step 0). |
| `missing checkpoint ... run python3 -m ml_2.training.train_live_checkpoints first` | Run step 2. |
| `channel MISMATCH: expected channel 6, got channel N` | Router drifted off channel 6. Fix the router -- don't `--force` past this, since both checkpoints were only ever trained on channel-6 data. |
| `bandwidth MISMATCH` | Router negotiated 40MHz instead of the expected 20MHz. Same advice: fix the router rather than `--force` unless you specifically want to test out-of-distribution behavior. |
| Calibration takes much longer than `--calib-seconds` | Packet rate is unusually low -- the script waits past the nominal duration until it has a full window's worth of packets (a safety floor, not a bug). Check `connect_check.sh` still passes and the stimulus traffic generator logged as started. |
| `connect_check.sh` fails | Same causes/fixes as `TESTING_RUNBOOK.md`'s troubleshooting table. |

---

## 7. What this is NOT

- **Not a replacement for the offline evaluation.** `ml_2/evaluation/results/DAY_PERSON_MODEL_ACCURACY.md`
  and `backtest_summary.csv` are the trustworthy, ground-truth accuracy numbers. This script is a live
  smoke test on top of already-measured models, not a new evaluation methodology.
- **Not a data-collection tool.** Nothing gets written to `data/` -- use `cli_collect.py` for that.
- **Not testing every model in `ml_2`.** Only `gnn_subcarrier` and `radar_cnn` have live checkpoints
  and a live-inference script today, per this round's explicit ask. Extending this to the other
  `ml_2` models (SVM, GBM, transformers, embeddings, receiver fusion, etc.) would need the same
  checkpoint-training + live-script pattern repeated for each.
