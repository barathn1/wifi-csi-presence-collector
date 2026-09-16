# Home Model v2 (Day3+Day4, RandomForest) — Live Testing Runbook

Tests `ml/models_out/home_model_v2.joblib` — the RandomForest clip-recipe person-identification
model (`ml/training/train_home_model.py`, trained on Day3 2026-09-15 + Day4 2026-09-16 pooled,
anjali vs barath). Both training days are native 128-subcarrier/channel-6/20MHz.

**This is a different, simpler pipeline than `ml/inference/live_infer.py`** (the WhoFi-transformer
system covered by `LIVE_INFERENCE_RUNBOOK.md` / `DAY3_CH6_RUNBOOK.md`). There is **no live streaming
and no calibration step** here: `score_live_session.py` scores a session file you've already
recorded, after the fact — it does not connect to the board itself or build an empty-room baseline.
If you want the real-time, calibration-based experience, use `live_infer.py` instead.

Run everything below from the repo root, with `.venv-ml` activated (or prefix commands with
`./.venv-ml/Scripts/python.exe` on Windows).

---

## 0. Preflight

Only one process can hold the ESP32 transport at a time — make sure nothing else
(`live_infer.py`, `ml.visualization.player_server`, another `cli_collect` run) is currently
connected. Then:
```
scripts/reconnect.sh
scripts/connect_check.sh
```
Both must pass (`connect_check.sh` should print `OK: board ... connected: ...`) before continuing.

**Confirm the router is still on channel 6 / 20MHz** before recording — this model was trained
exclusively on that band. Unlike `live_infer.py`, this pipeline has no automatic channel/bandwidth
gate, so a drift to a different channel (e.g. 11) won't be caught for you; check manually with
`scripts/check_bandwidth.sh` on a fresh recording if unsure.

---

## 1. The model is already trained

```
python -m ml.training.train_home_model
```
Only re-run this if you collect more Day3/Day4 data and want to fold it in — it overwrites
`ml/models_out/home_model_v2.joblib`. Already done once; skip unless you need to retrain.

**Honest expectations, stated plainly**: this project has never measured same-day accuracy above
~70–95% (varies a lot by motion) or cross-day accuracy above ~55–70%, using this same clip recipe —
see `ml/evaluation/results/experiment_log.csv` (`day3_*` and `tfmamba_approx_day3_*` rows) and
`train_home_model.py`'s own docstring. Day3→Day4 itself was never cross-day-validated before this
model shipped (v2 pools both days rather than testing one against the other). Treat today's live
result as a measurement, not a demo.

---

## 2. Record a live session

```
python -m collector.cli_collect --label authorized --person-id anjali --motion standing --duration 60
```
Swap `--person-id` for `barath`, `--motion` for `walking`, and `--duration` as you like (omit to stop
with Ctrl+C instead). This also works for an unauthorized/stranger test:
```
python -m collector.cli_collect --label unauthorized --person-id <name> --motion walking --duration 60
```
When it finishes, it logs the exact path, e.g.:
```
wrote 12345 samples to data/authorized/2026-09-16/20260916_HHMMSS_anjali -- avg 143.2 Hz over 60s (0 dropped, 0.0% loss)
```
Copy that path — you'll pass it to the scorer next. (This also auto-updates `data/manifest.csv`,
though that's not needed for scoring.)

---

## 3. Score the session

```
python -m ml.inference.score_live_session data/authorized/2026-09-16/20260916_HHMMSS_anjali --true-person anjali
```
- Drop `--true-person` if you'd rather just see the raw prediction with no accuracy comparison.
- `--true-person` should be the actual person who was recorded (or omitted for a genuine stranger —
  there's no "reject" option here, the model always outputs one of its two trained classes: anjali or
  barath; it does not tell you "neither").

Expected output:
```
loaded ml/models_out/home_model_v2.joblib (trained on ['2026-09-15', '2026-09-16'], created ...)
NOTE: Trained on Day3+Day4 (both native 128-sub/channel6/20MHz, no cross-mode padding needed). ...

37 clips scored:
  clip   0 (t~0s): predicted=anjali   confidence=0.71
  clip   1 (t~1s): predicted=anjali   confidence=0.68
  ...

majority vote over 37 clips: anjali ({'anjali': 30, 'barath': 7})

ground truth=anjali
  per-clip accuracy: 81.1% (30/37)
  session-level (majority vote) correct: True
```
- **Trust the majority vote over any single clip** — clips are 3-second, 1-second-stride windows, so
  individual predictions flicker; the session-level vote is the more stable read.
- If it prints `no usable clips`, the session was too short or every 3-second clip fell below the
  50%-packet-coverage threshold — record a longer session (recommend ≥30s).
- **Only ever outputs `anjali` or `barath`** — this model was trained as a closed 2-class problem, not
  an open-set detector. Testing with a genuine stranger will still force a prediction of one of those
  two names; that's expected behavior, not a bug, and isn't a meaningful "did it reject the stranger"
  test for this particular model (unlike the `taskD_auth_vs_nonauth` model in `live_infer.py`, which
  is explicitly trained to output "not authorized").

---

## 4. What to try

- **Same person, standing vs. walking** — this project's numbers so far suggest motion matters a lot
  (walking clips have scored notably higher than standing in several runs); worth comparing both.
- **Both anjali and barath**, more than one session each, to see if the result is consistent or
  session-dependent.
- **A genuine stranger** — remember it will still be forced into `anjali`/`barath`; the interesting
  question here is which one it defaults to and how confident it is, not whether it correctly says
  "neither."

---

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `connect_check.sh` fails | See `TESTING_RUNBOOK.md`'s troubleshooting table — same underlying connection `cli_collect.py` needs. |
| `no usable clips (session too short...)` | Record a longer session (≥30s recommended) or check the packet rate wasn't unusually low. |
| Predictions look random / low confidence throughout | Check the router hasn't drifted off channel 6 — this pipeline won't catch that for you (see step 0). |
| `FileNotFoundError` on the artifact | Run step 1 (`train_home_model.py`) first — `home_model_v2.joblib` must exist in `ml/models_out/`. |

---

## 6. What this is NOT

- **Not a real-time/streaming test.** Record first (step 2), then score after the fact (step 3) — there's
  no live "watch it happen" view. For that, use `live_infer.py` (see `LIVE_INFERENCE_RUNBOOK.md` /
  `DAY3_CH6_RUNBOOK.md`).
- **Not an open-set / intruder-rejection test.** This model only ever answers "anjali or barath,"
  never "neither" — it's a closed 2-class identification model, not an authorization gate.
- **Not calibrated against an empty-room baseline.** The clip-recipe features here are raw per-clip
  amplitude moments; there is no z-scoring against a `none` session the way `live_infer.py`'s `calibA`
  step does.
