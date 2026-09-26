# Live deployment guide

What the data actually supports for a real-time demo, and why. Read `FINDINGS.md` first for the
numbers this is built on.

## What this system can and can't do today

- **Can**: distinguish Anjali from Barath while one of them is walking, in real time, with increasing
  confidence the longer they walk. Confirmed decisively above chance, cross-day.
- **Can't yet**: reliably reject an arbitrary stranger as "not authorized" (see `FINDINGS.md` section
  2) -- accuracy there is statistically indistinguishable from chance. **Don't ship an auth/non-auth
  gate on this model.** If the demo needs to handle strangers, either (a) treat it as closed-set
  ("which of these 2 known people is walking" -- assumes it's always one of them), or (b) collect
  more sessions per stranger before trusting an open-set reject.

## Recommended architecture: presence gate -> silent accumulation -> reveal -> keep refining

```
 [CSI stream] -> [presence detector] -> (idle, show nothing)
                        |
                  presence=True
                        v
              [start accumulating windows, decode+clean+mask each one, run CNN+Attention,
               keep a running mean of P(barath) -- show NOTHING yet]
                        |
                elapsed >= ~30-45s of walking (see below)
                        v
              [reveal: "Anjali" / "Barath" based on running-mean P(barath) > 0.5]
                        |
                        v
              [keep updating the running mean + displayed confidence as more windows arrive,
               for as long as presence continues]
```

### Why gate on presence first

Presence detection (occupied vs. empty room) is a much higher-SNR, easier problem than identity --
77-96% accuracy under the same strict cross-day splits used for identity (`analysis/*_ch6`'s
`environment_control.py` / `csi_auth`'s equivalent positive control). Use it to decide WHEN to even
start the identity clock; don't run the identity model against empty-room noise.

### Why wait before showing a prediction, and how long

From `FINDINGS.md` section 1's live-demo timing table (chronological, no peeking ahead):

| elapsed walking time | CNN+Attention accuracy |
|---|---|
| ~4s | 82% |
| ~12s | 84% |
| ~18s | 90% |
| ~24s | 84% |
| ~36s | 91% |
| ~48-60s | 87-90% |
| ~80s | 95% |
| ~111s | ~100% (n=32, treat as encouraging not proven) |

The first ~10s is genuinely unreliable (weaker models sit at 60-66% there -- barely better than a
coin flip) and would undermine trust if shown. By ~18-24s, the best model is already in the 84-90%
range; by ~36s it's over 90%. Past that, gains keep coming for CNN+Attention specifically but flatten
for everything else.

**Recommendation: reveal the first prediction at 30-45 seconds of continuous walking after presence
is confirmed**, not 60s and not 10s:
- Long enough to be past the noisy opening window and solidly into the "the strong model is already
  right most of the time" zone.
- Short enough that the demo doesn't feel like it's stalling -- the accuracy gain from 45s to 60s is
  small, and from 60s to 90s is only really worth it for CNN+Attention.
- After the reveal, keep the running mean updating live rather than freezing it -- confidence should
  visibly climb if the person keeps walking, which is honest about what the model actually knows at
  each point and still captures the gains available out to ~60-90s.

If you'd rather keep it simpler than a continuously-updating display: reveal once at 30s, and
optionally re-reveal/upgrade the label once more at 60s if it disagrees (rare, but the table shows
short-window predictions do occasionally flip).

## Model choice for the demo

**Use CNN+Attention as primary.** It's the strongest or tied-strongest model at essentially every
checkpoint in the timing table, and its accuracy keeps improving the longest instead of plateauing at
~20s like the others. Use SVM as a fast, simple fallback/sanity-check display (its accuracy is
respectable and it's far cheaper to run than the deep models) but don't lead the display with it.

## Loading the checkpoints

`train_final.py` produces `checkpoints/{svm,cnn_bilstm,cnn_attention}_final.*`, trained on all 6 days
pooled (see `checkpoints/README.md` for exact provenance/metadata). These are NOT the leave-one-day-out
fold models used to produce the FINDINGS.md numbers -- they're one final version meant to actually run.

```python
import joblib
import torch
from models import CnnAttention  # or CnnLstm for the bilstm checkpoint

# --- classical: SVM (scaler bundled in the pipeline) ---
svm = joblib.load("checkpoints/svm_final.joblib")
proba_barath = svm.predict_proba(X_stats_window)[:, 1]   # X_stats_window: (n, 874) handcrafted features

# --- deep: CNN+Attention ---
ckpt = torch.load("checkpoints/cnn_attention_final.pt")
model = CnnAttention(n_subcarriers=ckpt["n_subcarriers"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

# normalize a raw amplitude window the SAME way training did, using the saved stats:
x_norm = (raw_amplitude_window - ckpt["sub_mean"]) / ckpt["sub_std"]
with torch.no_grad():
    proba_barath = torch.sigmoid(model(torch.from_numpy(x_norm))).numpy()
# ckpt["label_meaning"] == {0: "anjali", 1: "barath"}
```

Feature/window prep for a live window must exactly match training: decode the dominant `csi_len`
bucket (`decode.py`), Hampel-filter spikes (`cleaning.py`), drop the 19 null subcarriers
(`subcarrier_mask.py`), THEN either compute handcrafted stats (`features.py`, for SVM) or feed the
raw 200-packet x 109-subcarrier amplitude array directly (for the CNN models, after the saved
mean/std normalization above).

## Reference implementation (`live_inference.py` + `replay_demo.py`)

`live_inference.py` implements exactly the architecture above as a transport-agnostic state machine
(`LiveIdentitySession`) -- feed it one already-decoded, already-cleaned, already-masked window at a
time via `on_window(amplitude, phase, rssi, elapsed_s)`, it returns what should be on screen.
`replay_demo.py` drives it against a REAL recorded session (no hardware needed) to validate it
end-to-end: `python replay_demo.py <path-to-samples.npz> [reveal_after_s]`.

**Two real bugs building this that are worth knowing about, not just the design:**

1. **A single misclassified window is not a rare event -- debounce, don't trust one window.**
   The presence gate's headline 99.5% (`train_presence.py`) is a POOLED training-set number, not a
   per-window guarantee. Replaying a real empty-room recording window-by-window without any
   smoothing produced a false "presence confirmed" from literally the first window. Fix: require a
   rolling majority (4 of the last 5 windows) to agree before confirming OR dropping presence, not any
   single window's vote (`PRESENCE_VOTE_WINDOW`/`PRESENCE_VOTE_MIN_POSITIVE` in `live_inference.py`).

2. **Board consistency isn't optional, and multi-board session folders make it easy to get wrong.**
   2026-09-21 onward, every session directory has 3 files (one per receiver board). Loading the wrong
   one for a quick manual test produced a *sustained, high-confidence* (86-92%) false-positive across
   an entire ~4.5-minute empty-room recording -- not a flaky edge case, a systematically wrong answer,
   because the presence/identity models were trained exclusively on `ac:27:6e:a5:5b:c8`
   (`FIXED_BOARD` in `data.py`) and know nothing about the other two boards' noise floor. Any live
   feed MUST come from that specific receiver.

### Validated behavior (after both fixes, correct board)

| session | presence | reveal (at ~35s) | correct? |
|---|---|---|---|
| Anjali, walking | confirmed ~1s in | "anjali", 76.2% -> 74.5% | yes |
| Barath, walking | confirmed ~1s in | "barath", 84.6% -> 80.7% | yes |
| Empty room (~269s) | stays idle almost throughout; one brief flicker near the end correctly suppressed by the debounce before it could reveal | -- | yes |
| Stranger (abdul), walking | confirmed ~1s in | "barath", 74.5% -> 71.5% | **confidently wrong -- expected.** This is the open-set limitation from `FINDINGS.md` section 2, now visible in the live engine too: presence correctly fires on anyone, but the identity model is a strict binary classifier and WILL pick one of the two known people no matter who's actually there. Do not treat this output as authentication. |

## Open-set rejection: tested, doesn't work yet (`openset.py`)

The stranger-misclassified-as-Barath behavior above is real and reproducible. Two standard fixes were
tried and tested honestly (thresholds picked on half the strangers, evaluated on the OTHER, never-seen
half):

| approach | calibration AUC (known vs. calibration-strangers) | held-out stranger reject rate |
|---|---|---|
| Confidence-band (reject if P(barath) stays near 0.5) | 0.549 | 7.6% |
| Distance-to-centroid, in the CNN+Attention model's own learned embedding | 0.573 | 9.6% |

Both are barely above chance and reject under 10% of held-out strangers' windows -- **neither is
reliable enough to use.** This isn't a sign the wrong technique was tried; it's the expected failure
mode of a classifier trained only to distinguish two known people -- it never learned what "neither"
looks like, so it produces confident-looking outputs for any input, in-distribution or not. The
embedding-distance approach was specifically worth checking (a discriminatively-trained internal
representation could in principle separate better than raw features do), and it didn't.
**Conclusion: this needs more/better stranger data (ideally multiple sessions per stranger across
multiple days, matching how Anjali/Barath were collected), not a cleverer algorithm on the current
data.** See `openset.py` for the full calibration/evaluation code if more stranger data becomes
available -- the same script, just with a real calibration set, is the right next step.

## What would most improve this before a real deployment

1. **More non-auth sessions per stranger**, if open-set rejection is ever needed -- the current
   auth-vs-non-auth result is a data problem, not a modeling one (see `FINDINGS.md` section 2).
2. **More days.** 6 days is enough to show the identity signal is day-stable, not enough to rule out a
   ceiling -- `analysis/walking_ch6/FINDINGS.md` already showed the honest cross-day number sagged
   slightly (72% to 68% at window-level) going from 3 to 6 days; a 4th+ round of collection would
   clarify whether that's noise or a real trend.
3. **Fix the clock-reset bug** in the firmware/collector (6 of 44 sessions in this batch have a
   `device_time_us` discontinuity) -- doesn't affect correctness here, but would corrupt any
   real-time "how long has this person been walking" display without a guard.
