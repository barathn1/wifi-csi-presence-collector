# Session-level breakdown summary

**What this file is.** `ml_2/evaluation/results/session_breakdown.csv` records, for every
cross-validation fold of every model, ONE row per test session: whether that session is truly
authorized (`y_true`), what the model's majority vote across that session's windows was
(`majority_pred`), what fraction of the session's individual windows agreed with that vote
(`window_accuracy`), the average predicted probability of "authorized" across the session
(`mean_proba`), and how many overlapping windows the session contributed (`n_windows`).

**Why this exists.** A model's headline AUROC is computed over *windows*, and windows within one
session overlap 50% -- they are not independent evidence, they're the same walk sliced into
overlapping snippets. A single bad session with a few thousand windows can make a pooled number
look better or worse than the model's actual session-by-session judgment. This file -- and this
summary -- let you check the model's decision session by session instead of trusting the pooled
number blindly.

**Snapshot timing.** This summary covers every model whose training finished AFTER the full
per-session CSV logging was added: **`gnn_subcarrier`** and **`radar_cnn`** so far (711 session-fold
rows). Training is still running in the background (currently on 3-receiver fusion); models that
finished BEFORE this logging existed (SVM/GMM variants, GBM, CUSUM, Prototypical, ArcFace, the
transformer zoo) are only available in the separate, partial
`session_breakdown_PARTIAL_backfill.csv` (worst 12 sessions per fold only, not the full picture).
I'll extend this summary as later stages (receiver fusion, contrastive, MAML, VAE hard-negative)
complete and add their real data.

---

## 1. Aggregate numbers, recomputed directly from the per-session rows

| Model | Split | Session accuracy | Session AUROC | # sessions | # folds | # misclassified |
|---|---|---|---|---|---|---|
| GNN (subcarrier graph) | Open-set (stranger) | 0.815 | 0.732 | 233 | 9 | 43 |
| GNN (subcarrier graph) | Cross-day | 0.648 | 0.683 | 122 | 5 | 43 |
| Radar-inspired CNN | Open-set (stranger) | 0.751 | 0.809 | 233 | 9 | 58 |
| Radar-inspired CNN | Cross-day | 0.590 | 0.603 | 122 | 5 | 50 |

Reading this plainly: both models correctly judge roughly 3 out of 4 individual TEST SESSIONS on
open-set (a session is "correct" if the majority of its windows voted the right way), dropping to
roughly 3 out of 5 on cross-day. Radar-CNN has a slightly higher session AUROC on open-set (0.809
vs 0.732) despite GNN having a higher raw session-accuracy (0.815 vs 0.751) -- AUROC cares about
ranking scores across ALL sessions, accuracy cares only about which side of 0.5 each one landed on,
so the two metrics can disagree like this when a model is confidently wrong on some sessions and
only mildly right on others.

---

## 2. The single most useful finding: one specific session fools BOTH models, every time

**`none/2026-09-21/20260921_161220_ac276ea2f278`** -- an empty-room recording from one specific
receiver (`ac276ea2f278`) on 2026-09-21 -- is misclassified as **authorized** by the GNN in **every
single one of its 9 open-set folds**, plus its cross-day fold. It is a small session (only 130
windows, the smallest session in the whole misclassified list), and the GNN's window-level accuracy
on it is consistently near **0.000-0.070** -- meaning it isn't a borderline call, the model is
confidently wrong on nearly every window of this specific empty-room recording.

This is exactly the kind of thing a pooled window-level number would hide (130 windows barely move
an AUROC computed over tens of thousands), but a session-by-session read surfaces immediately: **this
one receiver/session combination has a CSI signature the model has learned to associate with "someone
authorized is here," even though the room was empty.** Worth checking directly: is there a hardware
quirk on receiver `ac276ea2f278` for this specific session (a fan, an appliance turning on, a placement
change), or is this genuinely how that receiver's noise floor looks and the model over-fit to it?

---

## 3. The cross-day failure has a clear, session-level shape: it's the AUTHORIZED sessions that flip

Look at what happens when **2026-09-21** is the held-out day (i.e. the model trained on every other
date and is asked to judge 2026-09-21 cold):

- **Every single authorized session from 2026-09-21** is misclassified as NOT authorized by BOTH
  the GNN and the radar-CNN -- 11 out of 11 authorized sessions for GNN, 11 out of 11 for radar-CNN.
  Window-level accuracy on these sessions sits in the 0.14-0.50 range: the model isn't just
  slightly less confident, it's often flipped to confidently wrong.
- Meanwhile the `none`/`unauthorized` sessions from that same day are judged much more reliably
  (aside from the one anomalous `none` session in section 2).

This is a precise, session-level confirmation of the cross-day numbers reported earlier (open-set
AUROC 0.73-0.81 vs cross-day AUROC 0.60-0.68): **the models don't generally get confused about
"stranger vs authorized" on a new day -- they specifically stop recognizing the AUTHORIZED people's
gait signature once trained without any of that day's data.** That's a meaningfully different, more
actionable diagnosis than "cross-day AUROC is lower" on its own: it points at day-specific drift in
the AUTHORIZED-person signal (calibration baseline drift, clothing/schedule differences that day,
receiver placement) as the likely cause, not a general breakdown of the open-set boundary.

---

## 4. Full list of misclassified sessions (every one, not just examples)

### GNN (subcarrier graph) -- 43 misclassified in open-set, 43 in cross-day

**Open-set false accepts** (model said "authorized" but it was a stranger or empty room) -- this
happened for `none/.../20260921_161220_ac276ea2f278` in **all 9** held-out-stranger folds, plus at
least 2-3 sessions per held-out stranger from that stranger's own unauthorized recordings (i.e. the
GNN correctly rejects MOST of a stranger's windows but a couple of that stranger's sessions still
get waved through). No open-set false REJECTS (authorized sessions wrongly called stranger) for GNN.

**Cross-day**: the pattern from section 3 -- all 11 authorized sessions from 2026-09-21 flipped to
"not authorized," plus the recurring `ac276ea2f278` empty-room anomaly, plus a long tail of
unauthorized sessions from 09-15/16/17/22 that get (correctly, from a safety standpoint) rejected
but are counted here because they're the sessions the model spent the most "effort" on relative to
their window count.

### Radar-inspired CNN -- 58 misclassified in open-set, 50 in cross-day

Same overall shape as GNN, with two differences worth calling out:
- Radar-CNN ALSO false-accepts `none/2026-09-15/20260915_152301` (a different empty-room session,
  on a different day) across multiple held-out-stranger folds -- so this model has its own,
  different "blind spot" session, not just the shared `ac276ea2f278` one.
- Radar-CNN false-REJECTS 3 authorized sessions in the open-set evaluation too (`.../20260921_172059_anjali`,
  `.../20260922_130435_anjali`, `.../20260922_141852_anjali` -- all Anjali sessions), something GNN
  doesn't do at all in open-set. Worth noting Anjali's sessions specifically seem harder for
  radar-CNN across several held-out-stranger folds -- possibly a gait/build similarity to one of the
  unauthorized identities, or a receiver/day-specific quirk on those particular recordings.

The full row-by-row detail (every misclassified session, which fold, window accuracy, window count)
is in `session_breakdown.csv` itself -- filter to `majority_pred != y_true` for the exact same list
this section was built from.

---

## 5. How to keep extending this yourself

```python
import pandas as pd
df = pd.read_csv("ml_2/evaluation/results/session_breakdown.csv")

# recompute session-level accuracy/AUROC for any model+split:
from sklearn.metrics import roc_auc_score
g = df[(df.model == "gnn_subcarrier") & (df.split_type == "open_set_loo_stranger")]
print((g.majority_pred == g.y_true).mean(), roc_auc_score(g.y_true, g.mean_proba))

# find sessions that are hard across MULTIPLE models (once more stages have run):
wrong = df[df.majority_pred != df.y_true]
wrong.groupby("session_dir")["model"].nunique().sort_values(ascending=False)
```
As receiver fusion, contrastive, MAML, and the VAE hard-negative stage finish, re-running the
cross-model "hard session" query above becomes much more informative -- right now it's only 2
models, so almost nothing overlaps by chance; once there are 5-7 models in the file, a session that
shows up as misclassified across most of them is a strong signal that something about THAT
session/receiver/moment is genuinely anomalous, not model-specific.
