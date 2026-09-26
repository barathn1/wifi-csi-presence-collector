# The "Home Model": WiFi-CSI Person Identification — Explainer

## TL;DR

We use a home WiFi router + an ESP32 as a passive sensor. The ESP32 captures **Channel State
Information (CSI)** — how the WiFi signal's amplitude changes as it bounces around the room —
and we train a classifier to recognize **who** is in the room from that signal alone, no camera,
no wearable, no phone. The current deployed model ("home model v2") is a **Random Forest**
trained to tell **anjali vs barath** apart, and it's right about **70–95%** of the time on
same-day data, dropping to **~55–70%** across different days. This document explains the full
pipeline: what data goes in, how it's turned into numbers a classifier can use, what the model
actually is, and where its honest limits are.

---

## 1. The problem

WiFi signals passing through a room get distorted by whatever's in that room — walls, furniture,
and people. A person's body absorbs and reflects the signal in a way that depends on their size,
posture, and movement. **CSI** is the raw per-subcarrier record of that distortion that a WiFi
radio already computes internally (it needs it to demodulate the signal correctly) — we're just
reading it out instead of throwing it away.

The bet: if two different people distort the signal in subtly different, *consistent* ways, a
classifier can learn to tell them apart just from that distortion pattern, over time, without any
camera or device on the person.

## 2. What the raw data looks like

Each WiFi packet the ESP32 hears produces one CSI reading: a complex number (real + imaginary
part) **per subcarrier** (the router splits its channel into ~128 narrow frequency slices called
subcarriers; each one gets distorted slightly differently by the room). So one packet gives you
a vector of ~128 complex numbers, and a session recording gives you a stream of these vectors
over time, at roughly 100–150 packets/second.

We convert each complex CSI value to a real number using its **amplitude** (how much the signal's
strength changed, ignoring the timing/phase of the wave):

```
amplitude = sqrt(real² + imaginary²)
```

Amplitude is used instead of phase because phase is very sensitive to small clock/sampling
imperfections on cheap radios like the ESP32 — it's noisy in a way that doesn't reflect the room,
just the hardware. Amplitude is the standard, more robust channel for this kind of sensing.

## 3. From raw packets to something a classifier can learn from

A raw CSI stream is a long, variable-length, noisy sequence — not directly usable by a classifier
that expects a fixed-size input. The pipeline turns it into fixed-size "clip" feature vectors in
five steps:

| Step | What happens | Why |
|---|---|---|
| **1. Clean** | Drop corrupted packets (flagged `first_word_invalid`) and duplicate-timestamp packets | Bad packets would corrupt any statistic computed from them |
| **2. Keep one packet shape** | A session occasionally has a few malformed packets with a different length; keep only the most common (dominant) length | Keeps the feature vector width consistent |
| **3. Slice into clips** | Cut the session into **3-second clips, sliding forward every 1 second** (so clips overlap by 2/3) | 3 s is long enough to gather ~100+ packets for stable statistics; the 1-second stride gives more, smoother training samples than non-overlapping clips, and lets a live system update its guess every second |
| **4. Quality gate** | Drop any clip that captured **less than 50%** of the packets it should have (based on the session's average packet rate) | If WiFi congestion or interference starves a clip of packets, its statistics become unreliable — better to skip it than train on garbage |
| **5. Summarize each clip** | For every subcarrier, compute 4 numbers over all the packets in that clip (see below) | Converts a variable-length clip into one fixed-length feature vector |

### The four numbers per subcarrier: statistical moments

For each subcarrier's amplitude values `x₁, x₂, ..., xₙ` within a clip, we compute:

- **Mean** — the average signal strength: `μ = (1/N) Σ xᵢ`
- **Standard deviation** — how much it fluctuated: `σ = sqrt( (1/N) Σ (xᵢ − μ)² )`
- **Skewness** — is the fluctuation lopsided in one direction or symmetric?: `(1/N) Σ (xᵢ−μ)³ / σ³`
- **Excess kurtosis** — are there occasional big spikes (heavy tails), or is it evenly spread?: `(1/N) Σ (xᵢ−μ)⁴ / σ⁴ − 3`

These are the classic first four "shape descriptors" of a distribution — they don't assume the
data follows any particular bell curve, they just describe its shape numerically. The intuition:
someone standing still produces a fairly steady, symmetric signal (low std, low skew); someone
walking produces bigger swings and occasional sharp spikes as their body moves through the
signal path (higher std, higher kurtosis).

**One clip → one feature vector** = `[all subcarriers' means, all subcarriers' stds, all
subcarriers' skews, all subcarriers' kurtoses]`, concatenated. With 128 subcarriers, that's
**4 × 128 = 512 numbers** describing one 3-second clip.

## 4. The model: Random Forest

We feed those 512-number vectors, each labeled with who was recorded, into a **Random Forest
classifier** — an ensemble of 300 decision trees.

**How a decision tree works, briefly**: it repeatedly asks yes/no questions like "is subcarrier
42's mean amplitude greater than 1,340?" and splits the training data accordingly, choosing at
each step the question that best separates anjali's clips from barath's. A single tree, grown
deep enough, will happily memorize quirks of its specific training data (overfit).

**How Random Forest fixes that**: instead of one tree, it grows 300 of them, and:
- each tree only sees a random resample of the training clips (some clips repeated, some left
  out) — called **bagging**,
- each tree, at each split, only gets to consider a random subset of the 512 features, not all
  of them,

so no two trees end up identical, and their individual mistakes tend to cancel out. The final
prediction is a **majority vote** across all 300 trees.

**Why this model and not a neural network**: across every architecture this project tried
(various transformer variants, different feature recipes), Random Forest was consistently the
most reliable — it matched or beat every deep-learning attempt, and unlike the transformer
models, it never collapsed into predicting only one person no matter the input. With only a few
thousand feature vectors to learn from (not millions), a simpler, more constrained model
generalizes better than a large neural network would.

**Settings used**: 300 trees, `class_weight="balanced"` (so the model isn't biased toward
whichever person happened to have more recorded clips), fixed random seed for reproducibility.

## 5. Training data

The deployed model ("v2") is trained on two full days of recordings:

| Day | Date | anjali clips | barath clips |
|---|---|---|---|
| Day 3 | 2026-09-15 | 1,854 | 1,615 |
| Day 4 | 2026-09-16 | 1,174 | 1,873 |
| **Total** | | | **6,516 clips** |

Both days used the same WiFi setup (channel 6, 20 MHz, 128 subcarriers), so no adjustment was
needed to combine them. Both "standing" and "walking" recordings are pooled together, since a
real live test won't tell the model in advance which one someone is doing.

An earlier version ("v1") was trained on two *different* days that used incompatible WiFi
settings (one had 186 subcarriers, the other 128) and needed extra padding to combine — it's kept
around but no longer the default.

## 6. Using the model live

1. Record a session the same way training data was recorded.
2. Run the same clip pipeline (3 s clips, 1 s stride, drop low-coverage clips, compute the same
   512 numbers per clip).
3. The model predicts a person for **each individual clip**, with a confidence score.
4. **Take the majority vote across all clips in the session** as the final answer — this is the
   number that matters, not any single clip's guess. Individual clips are short and noisy, so
   predictions flicker between the two people; averaging over many overlapping clips smooths
   that out, the same way averaging many noisy measurements gives a more stable estimate than
   trusting any one of them.

## 7. Honest limitations — read this before demoing it

- **It only knows two people: anjali and barath.** It is a closed, 2-class model. Show it a
  complete stranger and it will *still* confidently guess "anjali" or "barath" — it has no
  concept of "neither" or "unknown." That's not a bug, it's a fundamental limitation of this
  model type (see the separate open-set experiment below for the one attempt at fixing this).
- **Accuracy is motion- and day-dependent.** Same-day accuracy has ranged 70–95%; accuracy
  across different days has been measured as low as 43% and as high as 69–70% depending on the
  exact comparison. Walking tends to score better than standing.
- **This exact Day3→Day4 pairing was never validated before shipping.** The model pools both
  days rather than testing one against the other, so today's live accuracy is a genuine
  measurement, not a number we already know in advance.
- **No calibration step.** Unlike the more advanced streaming pipeline in this project
  (`live_infer.py`), this model does zero comparison against an "empty room" baseline — it works
  directly off the raw clip statistics.
- **Sensitive to WiFi channel.** It was trained exclusively on channel 6 / 20 MHz. If the router
  drifts to a different channel, nothing catches that automatically — predictions will just
  quietly get worse.

## 8. Why this matters / what's next

This closed 2-class result is really a stepping stone toward the more useful question: **can the
system say "I don't recognize this person" instead of forcing a guess?** That's a genuinely
harder problem (open-set recognition), and this project has a separate, more experimental attempt
at it (a transformer-based embedding model paired with an "Extreme Value Machine" for
accept/reject decisions) — early results there show it can reject strangers sometimes, but with a
meaningfully higher false-accept rate than we'd want for anything beyond experimentation. Happy
to walk through that one separately if useful.

---

## Appendix: key terms, plainly

| Term | Plain meaning |
|---|---|
| CSI (Channel State Information) | A record of exactly how a WiFi signal got distorted on its way from sender to receiver, broken down by frequency slice (subcarrier) |
| Subcarrier | One narrow frequency slice within the WiFi channel; ~128 of them here |
| Amplitude | Signal strength (ignores timing/phase) |
| Clip | A fixed 3-second slice of a recording, used as one training/prediction unit |
| Feature vector | The 512 numbers (mean/std/skew/kurtosis × 128 subcarriers) that summarize one clip |
| Random Forest | An ensemble of many decision trees whose votes are combined for a more reliable prediction |
| Majority vote | Taking the most common prediction across many clips as the final, trusted answer |
| Closed-set vs open-set | Closed-set: model must pick from a fixed list of known people. Open-set: model can also say "none of the above" |
