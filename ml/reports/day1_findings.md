# Day-1 findings: what this data can and can't tell you

Everything here is from `data/{authorized,unauthorized,none}/2026-09-09` -- **one day, no cross-day
data yet**. Every number below is same-day. See `ml/README.md` for how to reproduce, and
`RESEARCH_NOTES.md` for the literature these numbers are being compared against.

## TL;DR -- auth vs non-auth first, since that's the actual point of this project

**Want to see noise vs. signal directly, per subcarrier?** Jump to "Signal vs. noise, made visible"
below -- it builds an empirical noise floor (shuffle session labels 5000 times, see what effect sizes
pure chance produces) and shows that **authorized vs unauthorized currently has 0/186 subcarriers that
clear it -- every visible difference is within the range randomly-relabeled data would also produce.**
`unauthorized vs none` is the only comparison with real, visible signal (20/186 subcarriers), and even
that one is suspect (see the time-of-day confound note in that section).

**Headline finding: the model almost never mistakes an empty room for the authorized person (0-5%
false-accept), but it lets a real unauthorized person through as "authorized" 25-44% of the time**,
depending which sessions are held out. A single blended accuracy/AUROC number for "authorized vs
everything else" hides this completely -- it looks decent (66% accuracy, AUROC ~0.79) only because
empty-room negatives are so easy that they drag the average up. Broken down by what the negative
actually was (RandomForest, session-disjoint folds, `taskD_auth_vs_nonauth`):

| Fold | unauthorized false-accept rate | none false-accept rate |
|---|---|---|
| 1 | 43.7% | 4.6% |
| 2 | 35.4% | 0.9% |
| 3 | 25.2% | 0.0% |
| 4 | 40.8% | 0.0% |

**This is the real number for "can I classify authorized vs unauthorized": no, not reliably yet** --
roughly a third of the time, a genuine unauthorized person is classified as authorized. If this were a
real access-control gate, that's the number that matters, not the blended 66-79%. See "The central
question" section below for the full picture (RandomForest, transformers, and the proper open-set
embedding model, all evaluated the same way).

**One genuinely good lever: don't decide from a single ~1-second window.** Averaging a model's score over
~5-10 seconds of consecutive windows (same trained model, just more context before deciding) nearly
halves the unauthorized-false-accept rate, from 32.6% (1 window) to 18.4% (20 windows, ~10s) -- see
"Deciding from more than one window helps, cleanly" below. This is the cleanest positive result in the
whole report.

**"Can I differentiate between the 2 authorized people (anjali vs barath)?"** Not reliably, and the
reason why is itself the finding. The honest, session-disjoint 5-fold accuracy averages 48.3%
(coin-flip), but that average hides something more specific than "no signal": two folds score 82-88%,
two folds score 2-3% (worse than guessing), one scores 66%. A model that's sometimes very right and
sometimes very *wrong* about a binary label isn't measuring noise -- it's measuring a confound. anjali's
4 sessions are 2 walking/2 standing; barath's are 3 walking/1 standing. The classifier looks like it's
partly learning "walking vs standing," and it inverts completely whenever a held-out session breaks that
pattern. **Verdict: not yet answerable with this data as collected** -- motion type and identity are
entangled for these 2 people.

**What Day 1 data is good for**: presence detection (empty room vs. occupied) is the strongest signal
by far -- but even that ranges from 57.6% to 99.9% depending which sessions get held out, so "it works"
needs the caveat "on most, not all, held-out sessions."

## Signal vs. noise, made visible (`ml/visualization/signal_vs_noise.py`)

The most direct way to answer "what's noise and what's real": build an empirical noise floor by
shuffling which SESSIONS are labeled authorized/unauthorized/none 1000 times (keeping group sizes
fixed), recomputing each subcarrier's Cohen's d every time, and plotting the real effect size against
the band of effect sizes that shuffled (fake) labels produce. Anything inside the band is
indistinguishable from noise; anything poking out, with FDR correction across all 186 subcarriers, is
real. (Permuting whole *sessions*, not individual windows, matters: windows overlap 50% and many come
from the same session, so they aren't independent trials -- a per-window shuffle would report fake
near-certain "significance" everywhere from pure pseudo-replication.)

| Comparison | Amplitude: subcarriers that clear the noise floor | Phase |
|---|---|---|
| authorized vs unauthorized | **0 / 186** | 0 / 186 |
| authorized vs none | **0 / 186** (a peak around subcarrier ~127-135 visibly pokes above the band, d~1.3, but doesn't survive FDR correction across all 186 -- checked at both 1000 and 5000 permutations, same result, so it's not a resolution artifact) | 0 / 186 |
| unauthorized vs none | **20 / 186** (subcarriers ~125-145, d up to 1.6) | 1 / 186 |

(numbers above are from 5000 permutations, run twice at 1000 and 5000 to rule out the FDR threshold
simply being unresolvable at low permutation counts -- it wasn't the binding constraint here.)

See `signal_vs_noise_amp_authorized_vs_unauthorized.png`: the real per-subcarrier effect size line sits
entirely inside the gray noise band, everywhere, for both amplitude and phase -- **by this test, there
is currently no subcarrier where authorized vs unauthorized (or authorized vs none) differs by more than
session-to-session noise alone.** Compare to `signal_vs_noise_amp_unauthorized_vs_none.png`, where a
cluster of subcarriers spikes dramatically above the band (visibly, not just statistically -- see the
companion `distributions_amp_unauthorized_vs_none.png` violin plots: unauthorized sessions sit
completely above `none` sessions' distribution for those subcarriers, no overlap). `distributions_amp_
authorized_vs_unauthorized.png` shows the opposite: even the 4 "best" subcarriers by raw effect size are
heavily overlapping violins with p-values around 0.06-0.08 -- close, but not real by this test.

**Read this carefully, it does not mean the earlier classifiers were fake.** This test only checks ONE
subcarrier's average value at a time; RandomForest/transformers combine ~1490 features (all subcarriers
x mean/std/skew/kurtosis x amplitude/phase) jointly, and a real multivariate pattern can exist even when
no single univariate feature clears a strict per-feature bar. What this test *does* show is that the
signal, if it exists for authorized-vs-unauthorized, is weak and diffuse across many correlated features
rather than concentrated in a few strong subcarriers -- consistent with the earlier finding that
per-window classifiers swing wildly across folds (task0: 57.6-99.9%, taskD false-accept: 25-44%) instead
of hitting a stable number. **Also worth real suspicion**: the one comparison that DID clear the bar
(unauthorized vs none) is exactly the pair most confounded by time-of-day -- every unauthorized session
was collected in one late-day block, while `none` sessions were mostly earlier (see "time-of-day
confound" in the original exploration). That 24-subcarrier "signal" may partly or entirely be ambient
drift over the ~2.5 hour collection window, not a real occupied-vs-empty effect. This needs Day 2 data
(a second day's unauthorized-vs-none comparison, collected without the same time-block confound) before
trusting it as real.

## Methodology note: two different rigor levels below

- **Stage 1 (RandomForest baseline)**: full session/person-disjoint **5-fold CV** -- the trustworthy
  number, reported as mean +/- fold spread.
- **Stage 2 (transformer/CNN/LSTM zoo)**: a **single** 80/20 session-disjoint split, for CPU-time budget
  reasons (see `ml/training/train.py`'s docstring). By coincidence it's the *same* fold as Stage 1's
  fold 0 -- which Stage 1's own fold breakdown shows is an unusually **hard** fold for task0 and an
  unusually **easy** fold for taskB. Read Stage 2's numbers as "does this architecture work at all and
  roughly how does it compare to the others on this one split," not as a final ranking. Full k-fold CV
  for every architecture is the natural next step once there's a reason to trust one architecture enough
  to invest the compute.

## The central question: authorized vs non-authorized, every way we asked it

Four different framings of the same underlying question, from loosest to strictest. They tell a
consistent story once you look past the headline number of each: **decent-looking aggregate metrics
that are actually driven by trivially-easy empty-room rejection, sitting on top of a real
unauthorized-person false-accept rate that stays in the 25-44% range no matter which model is used.**

### taskD: authorized vs everything else (unauthorized OR none) -- the production framing

RandomForest, full session-disjoint 5-fold CV (fold 0 excluded -- it happened to hold out zero
authorized sessions, leaving nothing to score AUROC/EER against):

| Fold | Accuracy | EER | AUROC | false-accept: unauthorized-as-authorized | false-accept: none-as-authorized |
|---|---|---|---|---|---|
| 1 | 63.2% | 40.9% | 0.697 | **43.7%** | 4.6% |
| 2 | 84.4% | 15.7% | 0.916 | **35.4%** | 0.9% |
| 3 | 68.8% | 30.5% | 0.787 | **25.2%** | 0.0% |
| 4 | 69.1% | 31.2% | 0.765 | **40.8%** | 0.0% |
| mean | 66.2% | 29.6% | 0.791 | ~36% | ~1.4% |

Read the last two columns, not the aggregate ones: **an empty room is (almost) never mistaken for the
authorized person, but a real unauthorized person is, more than a third of the time on average.** The
blended 66% accuracy / 0.79 AUROC looks respectable specifically because `none` windows (roughly a third
of the negative class) are so easy that they pull the average up and hide how bad the model still is at
its actual job: telling the authorized person apart from an unauthorized one. This is the single most
important number in this whole report.

### Deciding from more than one window helps, cleanly

Every number above decides from a single ~1-second (200-packet) window. Per your instinct that a single
short window is a thin basis for a decision: `run_taskD_segment_aggregation.py` trains one
crossattn_transformer (fixed seed) and re-scores the SAME test windows three ways -- individually, and
averaged into ~5s (10 windows) and ~10s (20 windows) segments. Because this only changes how already-computed
scores are aggregated (not what's trained), it's a much lower-noise comparison than the architecture
comparison below:

| Decision horizon | Accuracy | AUROC | false-accept: unauthorized | false-accept: none |
|---|---|---|---|---|
| Single ~1s window | 66.0% | 0.729 | 32.6% | 1.6% |
| ~5s (10 windows averaged) | 69.0% | 0.746 | 21.8% | 0.0% |
| ~10s (20 windows averaged) | 70.1% | 0.752 | **18.4%** | 0.0% |

Monotonic improvement on every metric as the decision horizon grows, and the security-relevant number --
how often a real unauthorized person is accepted as authorized -- nearly halves (32.6% -> 18.4%) just
from watching for 10 seconds instead of 1, with the identical trained model. This is the single cleanest
positive result in this report, and it matches ARGUS's own finding that segment-level aggregation is a
"free" gain. **Practical implication for deployment: don't make an access decision from one window's
worth of data -- accumulate several seconds of consecutive windows first.**

### Architecture comparison: cross-attention vs late-fusion (results corrected below)

Two transformer models retrained on fold 1 (`run_taskD_focus.py`), same breakdown, looked very
promising at first:

| Model | Accuracy | EER | AUROC | false-accept: unauthorized | false-accept: none |
|---|---|---|---|---|---|
| RandomForest (fold 1) | 63.2% | 40.9% | 0.697 | 43.7% | 4.6% |
| dualbranch_transformer (ESP32-paper replica, fold 1) | 64.4% | 38.2% | 0.697 | 24.8% | 6.4% |
| crossattn_transformer (own idea, fold 1, run #1) | 69.3% | 35.4% | 0.757 | 26.2% | 3.7% |

That looked like a real win for the cross-attention idea -- until it was checked across the other 3
valid folds (`run_taskD_crossattn_cv.py`), which **also re-trained fold 1 from scratch as part of the
same sweep**:

| Fold | Accuracy | EER | AUROC | false-accept: unauthorized | false-accept: none |
|---|---|---|---|---|---|
| 1 (re-run) | 63.2% | 37.9% | 0.698 | **57.5%** | 10.0% |
| 2 | 71.7% | 28.3% | 0.798 | 55.0% | 2.3% |
| 3 | 67.1% | 39.8% | 0.739 | 22.1% | 0.0% |
| 4 | 62.0% | 37.6% | 0.712 | 63.6% | 0.4% |
| **mean** | 66.0% | 35.9% | **0.737** | **49.5%** | 3.2% |

**Fold 1, re-trained from scratch, gave AUROC 0.698 and a 57.5% unauthorized-false-accept rate -- not
the 0.757 / 26.2% from the first run.** Nothing about the data or split changed between those two runs;
only the random weight initialization and minibatch shuffling did (no seed was fixed at the time). On a
dataset this small, with only 4 epochs, **seed variance alone can be as large as the effect being
measured.** Averaged properly across folds, crossattn_transformer's mean AUROC (0.737) is actually
*below* RandomForest's (0.791) on the same 4 folds, and its mean unauthorized-false-accept rate (49.5%)
is *worse* than RandomForest's (36.3%). **The apparent win was noise, not signal** -- this is now fixed
going forward (`train.py` takes a `seed` argument), but every other Stage 2 single-run number in this
report should be read with the same caution until it's re-checked across seeds, not just across folds.
This correction is itself the most important methodological finding in this report: the "single split"
caveat wasn't sufficient by itself -- a fixed split still needs multiple seeds before trusting a
neural-net comparison.

### taskA: 3-way authorized/unauthorized/none (closed-set)

Session-disjoint 5-fold: **62.4%** (fold range 40.6-75.9%, std 12.1pp) vs. 33% chance. Real signal, but
a 3-way framing doesn't answer "should this person be let in" directly the way taskD does -- kept here
mainly as a sanity check that the classes are separable at all (they are, weakly).

### taskC: authorized vs unauthorized ONLY (excludes `none`) -- closed-set proxy for open-set

The narrowest framing: given that someone is present, is it the authorized person or not, with no
"nobody's there" option to lean on. Session-disjoint, leave-one-unauthorized-person-out (a genuinely
novel intruder each fold): **AUROC ~0.60 on average (range 0.44-0.72)**, sometimes *below 0.5* --
worse than a coin flip for one held-out person (kishore). This is the closed-set number; a plain
classifier structurally cannot say "unknown," it can only compare the two classes it was shown.

### The proper open-set model: embedding + cosine similarity + EER (not a closed-set proxy)

Per RESEARCH_NOTES.md/WhoFi/ARGUS, the right tool for "is this one of my 2 enrolled people, y/n" is an
embedding model with a tuned similarity threshold, not a closed-set classifier. Trained a WhoFi-style
model (contrastive loss, 4 epochs) per held-out unauthorized person, enrolled anjali+barath as centroid
embeddings from training data, scored every test window by cosine similarity:

| Held out | EER | AUROC | Held out | EER | AUROC |
|---|---|---|---|---|---|
| divya | 45.5% | 0.570 | sumanth | 43.6% | 0.553 |
| harshitha | 49.6% | 0.508 | vasu | 33.8% | **0.739** |
| kishore | 45.9% | 0.575 | vishwa | 38.3% | 0.689 |
| manas | 60.4% | 0.356 | **mean** | **45.7%** | **0.569** |
| sai | 50.1% | 0.490 | | | |
| siva | 43.9% | 0.639 | | | |

The proper embedding approach (mean AUROC 0.569) did **not** clearly beat the naive closed-set proxy
(mean AUROC ~0.60) here -- both land in the same weak-to-moderate band. This is very likely a training
budget artifact (4 epochs, no augmentation, no batch sampler for identity diversity -- all things
WhoFi's paper describes as load-bearing) rather than evidence the embedding approach is wrong; the
closed-set-vs-embedding comparison needs a fairer fight (more epochs, WhoFi's augmentation recipe)
before concluding anything about which approach is actually better on this hardware.

## Other tasks (secondary context)

| Task | Honest (session/person-disjoint) | Naive random split (leakage) | Gap |
|---|---|---|---|
| `task0_presence` (empty vs occupied) | **88.5%** (fold range 57.6-99.9%, std 15.8pp) | 99.8% | 11.3pp |
| `taskB_identity` (anjali vs barath) | **48.3%** (fold range 2.1-88.5%, std 38.4pp) | 91.5% | 43.2pp |

The naive-random-split column is deliberately kept in the log (`experiment_log.csv`, `split_type=
naive_random_LEAKY`) as a warning, not a result to use -- it's inflated by 11 to 43 points depending on
the task, exactly the failure mode RESEARCH_NOTES.md's literature review warns about repeatedly. The
worse the honest number, the bigger the gap -- taskB's "91.5% accuracy" from a random split is
*completely fake*; the model can't actually tell the two people apart (48.3%, confound-driven).

**Calibration ablation (Variant A, empty-room self-calibration) came back byte-identical to raw for
every task.** This isn't a bug or a null result about calibration -- it's a property of tree models:
RandomForest splits are invariant to any monotonic per-feature transform, and z-scoring before computing
mean/std/skew/kurtosis is exactly such a transform (skew/kurtosis are dimensionless and literally
unchanged by it; mean/std transform affinely). **The calibration question can only be tested on the
neural models**, where feature scale actually matters to gradient descent.

## Stage 2 results (transformer/CNN/LSTM zoo, single split)

| Model | task0 (hard fold) | taskA (fold0) | taskB (easy fold) |
|---|---|---|---|
| RandomForest (Stage 1, same fold, for reference) | 57.6% | 40.6% | 88.5% |
| cnn1d | 56.8% | 37.8% | 77.1% |
| bilstm | 57.6% | 40.5% | 81.7% |
| whofi_transformer (amplitude-only) | 57.1% | 37.7% | 61.3% |
| dualbranch_transformer (ESP32-paper replica) | 57.7% | **41.6%** | **88.7%** |
| crossattn_transformer (own idea) | **57.9%** | 35.7% | 82.1% |
| calib_context_transformer (own idea) | 57.6% | 30.1% (below chance) | 83.8% |

(taskD isn't in this table -- it's covered above in "The central question" with its own dedicated
comparison, since accuracy alone hides the false-accept breakdown that matters for that task.)

Takeaways, held to the single-split caveat above:
- **task0 is architecture-independent**: every model, including RF, lands in a tight 56.8-57.9% band on
  this specific held-out fold. That's strong evidence the difficulty is a property of *which sessions*
  got held out, not a modeling shortfall -- consistent with Stage 1's own 57.6-99.9% fold spread.
- **The dual-branch late-fusion replica (ESP32 paper architecture) was the strongest transformer on 2 of
  3 tasks**, matching RandomForest's ceiling on taskB (88.7% vs 88.5%) on this fold.
- **The cross-attention idea (own architecture) did not beat late-fusion concatenation anywhere, once
  checked properly.** It lost to plain late-fusion on task0 (+0.1pp, noise-level) and taskA (-5.9pp). On
  taskD it *looked* like a clear win in a single run (+6pp AUROC over RandomForest) -- full 4-fold CV
  with a fixed seed showed that win doesn't replicate; see "The central question" above for the full
  story and the seed-variance lesson it surfaced. Honest conclusion: **no evidence so far that
  cross-attention fusion beats simple concatenation** at this model scale and training budget. Worth
  retrying with more epochs and averaged seeds before concluding it never helps -- but the current
  evidence doesn't support it.
- **The calibration-context idea (own architecture) was the weakest on taskA** (30.1%, below the 33%
  chance floor) and mid-pack elsewhere. Expected: on Day-1-only data the "context" the model
  cross-attends to is a single constant vector (there's only one day's empty-room baseline), so it can't
  carry discriminative information yet -- it's just extra unused capacity. This idea genuinely needs
  Day 2 data (a *second*, different context vector) before it can show anything.
- **WhoFi's amplitude-only single branch was consistently the weakest transformer**, most sharply on
  taskB (61.3% vs. 82-89% for the dual-branch models). Some of this is an unfair comparison (it also has
  roughly half the parameters, having only one branch), but it's at least a data point against blindly
  trusting the "amplitude alone is nearly as good as amplitude+phase" finding from ARGUS/Trans-MAML
  (both multi-antenna-adjacent setups) on this single-antenna ESP32 identity task specifically.

## The motion-type confound, in detail

anjali: 2 walking + 2 standing sessions. barath: 3 walking + 1 standing session. This is the single
biggest data-quality issue this analysis found. It doesn't just weaken the identity signal -- it
actively **inverts** predictions on 2 of 5 folds (2-3% accuracy is not "no signal," it's confidently
wrong). Two ways to fix it going forward:
1. **Collection fix (best)**: collect balanced motion type per authorized person (equal walking/standing
   session counts) so the confound can't arise in the first place.
2. **Evaluation fix (works on existing data)**: re-run taskB restricted to standing-only or walking-only
   windows separately. Not done in this pass (small samples once split by motion **and** held out by
   session -- barath has only 1 standing session total), but should be the very next experiment before
   trusting any anjali-vs-barath number, positive or negative.

## Visualizations (`ml/visualization/figures/`)

- `embedding_by_class.png`: UMAP of window statistics, colored by authorized/unauthorized/none.
  `none` (empty room) forms clean, separate clusters; authorized and unauthorized are heavily
  intermixed in the same region -- a visual match for task0's strong signal and taskA/taskC's weak one.
- `embedding_identity.png`: same projection, anjali vs barath. Partial separation at the extremes, heavy
  overlap in the middle -- consistent with the fold-dependent accuracy above, not a clean two-cluster
  picture.
- `effect_size_*.png`: Cohen's d per subcarrier. `authorized vs unauthorized` is visually flat
  (near-white) compared to the strong blue/red bands for `* vs none` -- presence is separable, identity
  and authorization are not, and you can see it directly rather than inferring it from an accuracy
  number.
- `feature_importance_*.png`: RandomForest importances agree with the effect-size heatmaps on which
  subcarrier bands matter, and independently confirm amplitude dominates over phase for every task --
  two different methods agreeing is a real signal, not a modeling artifact.
- `attention_*.png`: temporal (not spectral) attention patterns from the trained cross-attention
  transformer -- which moments within the 1-second window the model leans on, per identity. The pattern
  found is vertical banding (attention depends almost entirely on WHICH key packet, barely on which
  query packet) -- the model has learned a handful of consistently-informative moments in the window
  rather than query-position-relative relationships. Only `barath` was covered by this run (the fold
  used happened to hold out only his sessions for test); re-run on a different fold for `anjali`'s
  pattern.

## Recommendations before trusting any number here as "the" answer

1. **Every neural-net comparison in this codebase needs multiple seeds, not just multiple folds** --
   `train.py` now takes a `seed` argument (added after fold 1 alone gave a 0.757 vs 0.698 AUROC swing
   from re-training with a different seed). Re-run the Stage 2 sweep and taskD experiments averaged over
   3-5 seeds before trusting any architecture ranking, including RandomForest-vs-transformer comparisons.
2. **RandomForest is currently the best-validated model for taskD** (mean AUROC 0.791 across 4 proper
   folds) -- not because it's architecturally superior, but because it's the only model evaluated with
   enough rigor (full CV, deterministic) to trust yet. Treat this as the baseline to beat, not the ceiling.
3. **Deploy segment-level aggregation (~5-10s), not single-window decisions** -- the one clean, low-noise
   win in this report (32.6% -> 18.4% unauthorized-false-accept, same trained model, just more context
   before deciding). Cheap to add to any of the other findings here; do this regardless of which
   architecture or task framing ends up winning.
4. **Never report a single blended accuracy/AUROC for "authorized vs non-authorized" again** -- always
   break it down by what the negative actually was (empty room vs real unauthorized person), the way
   `run_taskD_focus.py` does. The blended number is actively misleading here: it looks like a B+ (66-79%)
   while the actual security-relevant failure rate (a real intruder waved through as authorized) is
   25-44%.
5. **Give the embedding/EER approach (Task C) a fairer shot** -- more epochs, WhoFi's augmentation
   recipe (Gaussian noise, amplitude scaling, time shift), and a batch sampler for identity diversity,
   before concluding it doesn't beat the closed-set proxy. 4 epochs with none of that isn't a fair test.
6. **Collect Day 2 with motion-balanced authorized sessions** -- this single change would let the
   identity question (taskB) actually be asked cleanly for the first time.
7. **The empty-room calibration hypothesis (both variants) is still untested in any way that matters**
   -- Variant A needs a neural model with real CV (not RF); Variant B needs Day 2 to exist at all.
8. Day 3, whenever collected, must stay untouched until final evaluation -- don't fold it into any of
   the above, per RESEARCH_NOTES.md's repeated warning about pooled-random splits.
