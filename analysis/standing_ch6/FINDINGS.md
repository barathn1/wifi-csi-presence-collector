# Person-identifiability from standing CSI, channel-6 data -- findings

## Scope of this pass

- **Motion**: `standing` only (no gait/Doppler signal available to lean on -- the harder, "static body,
  multipath-only" version of the identification question).
- **Channel/hardware**: only the three dates that used 3 boards on WiFi channel 6 / 20MHz
  (`2026-09-21`, `2026-09-22`, `2026-09-24`) -- earlier dates used a different channel/bandwidth and
  would confound "person" with "PHY config" if pooled in. Every session read from the ONE receiver board
  present on every date (`ac:27:6e:a5:5b:c8`), so board/viewpoint is held fixed too.
- **People**: `anjali` (8 sessions) and `barath` (7 sessions) are the only two with enough sessions
  spread across enough days to test cross-day generalization at all. Strangers (5 people, 1 session
  each, only on 09-21/09-22) and 9 empty-room sessions are used as context/controls, not as a
  multi-way identification target.
- **Pipeline**: reuses the repo's existing decode/window/feature code (`ml/data_pipeline/*`) unchanged --
  200-packet (~1s) windows, 50% overlap, resampled to a shared 128-subcarrier grid, handcrafted
  mean/std/skew/kurtosis-per-subcarrier features (amplitude + phase) + RSSI stats. All code for this
  pass lives in `analysis/standing_ch6/`.

## 1-2. Raw signal & heatmaps (`raw_signal_plots.py`, `heatmaps.py`)

- `figures/raw_compare_identities.png`, `figures/heatmap_identity_comparison.png`: **a strong
  vertical-stripe amplitude pattern sits in the same subcarrier bands (~27-37 and ~93-99 are silent/zero,
  ~64-96 swings hardest) in every panel -- Anjali, Barath, every stranger, AND every empty-room
  recording.** That rules out those bands as a person signature by construction: they're identical when
  no one is even in the room, so they're a receiver/antenna/frame-format artifact, not a body effect.
- `figures/raw_mean_spectrum_profile.png`: the whole-session mean amplitude-vs-subcarrier curve is
  nearly identical across ALL sessions of all 4 identity groups -- **occupied sessions (any person)
  separate cleanly from empty-room (gray) on this view, but Anjali/Barath/stranger curves sit in one
  indistinguishable bundle.** This is the presence signal, not a person signal.
- `figures/heatmap_identity_comparison_residual.png`: subtracting each session's own per-subcarrier mean
  to strip the static gain pattern doesn't reveal a cleaner per-person structure either -- the dominant
  bands are actually bimodal (near-0 / near-saturation), so the "residual" just saturates the color scale;
  the calmer subcarrier ranges (0-27, 37-64, 99-128) show mild temporal texture but nothing that reads as
  person-specific at a glance across `figures/heatmap_grid_anjali.png` vs `figures/heatmap_grid_barath.png`.
- **Reading**: nothing in the raw amplitude, viewed by eye, points at a stable per-person visual
  signature. The strongest visible structure is a hardware artifact and a presence-vs-empty effect,
  both person-independent.

## 3. Feature-space separability (`feature_space.py`)

- `figures/feature_space_by_identity.png`: PCA and UMAP on the standardized 1026-dim handcrafted feature
  vector show **no visible clustering by person** -- Anjali/Barath/stranger/empty-room all sit in one
  overlapping cloud in both projections.
- `figures/feature_space_by_day.png`: coloring the same embedding by **day** shows the same thing --
  no day clustering either.
- `figures/feature_space_within_anjali_by_session.png` / `..._barath...`: restricting to ONE person and
  coloring by session shows that even a single person's own different recordings don't form tight,
  separable session blobs in 2D.
- **Reading**: at the coarse, linear/2-manifold level neither person identity nor day/session is the
  dominant source of variance in this feature space. Any signal here (if present) is subtle and
  multivariate, not visible in a 2D projection -- which is exactly why the quantitative tests below
  matter more than eyeballing a scatterplot.

## 4-5. Distance-based similarity (`distances.py`)

Pairwise Euclidean distance between standardized feature vectors, same-person vs different-person pairs,
split by time scope (`cache/distance_separability_by_level.csv`):

| scope | same-person mean dist | different-person mean dist | separability (ROC-AUC) |
|---|---|---|---|
| within-session | 38.01 | 39.39 | **0.545** |
| cross-session, same day | 39.10 | 39.39 | **0.499** (chance) |
| cross-day | 39.66 | 39.84 | **0.515** |
| **overall** | 39.31 | 39.70 | **0.516** |

(0.5 = distance carries zero information about same/different person; 1.0 = perfectly separable.)

- Even the best-case scope (within the same session, where two windows of the same person are compared)
  is only marginally separable (AUC 0.545). Cross-session-same-day is exactly chance. Cross-day is barely
  above chance.
- The KS test on the overall same- vs different-person distributions IS statistically significant
  (p=3e-23), but that's a large-sample-size artifact of ~90k pairs, not a usable effect -- the AUC (the
  actual effect size) never clears ~0.55 in any scope. **Figure `figures/distance_by_scope.png`
  confirms this visually: the two histograms are essentially the same curve in all three panels.**

## 6. Environment / positive control (`environment_control.py`)

To rule out "the pipeline just can't detect anything," the same features/splits/models were used for
presence detection (occupied vs empty room) instead of person ID:

| split | model | accuracy |
|---|---|---|
| cross-session 5-fold | svm_rbf | 0.936 |
| cross-session 5-fold | random_forest | 0.927 |
| leave-one-day-out | svm_rbf | 0.834 |
| leave-one-day-out | knn | 0.811 |
| leave-one-day-out | random_forest | 0.771 |

Presence is detected well within-day-family (0.93+) and reasonably even cross-day (0.77-0.83). **This
pipeline clearly can pick up a real, strong physical effect under the exact same strict splits used for
person-ID below -- so the weak person-ID result is not an artifact of broken features or code, it's
specific to person identity.**

Limitations of the environment control in this pass: receiver board and WiFi channel/bandwidth are held
fixed by construction (not tested as a variable); "same person, different body position" could not be
tested -- the collected metadata only distinguishes `standing` vs `walking`, not orientation/position
within standing.

## 7-9. Quantitative identity test + permutation control + leave-one-session/day-out (`baselines.py`)

kNN / RBF-SVM / RandomForest, anjali-vs-barath, real labels vs session-level label-shuffle control, on
THREE strict split types (`cache/baseline_results.csv`):

| split | model | **real** accuracy | **shuffled-label** accuracy |
|---|---|---|---|
| cross-session (5-fold GroupKFold, 3 usable folds) | knn | 0.644 | 0.467 |
| cross-session | random_forest | 0.676 | 0.443 |
| cross-session | svm_rbf | **0.708** | 0.474 |
| leave-one-day-out (3 folds) | knn | 0.548 | 0.459 |
| leave-one-day-out | random_forest | 0.561 | 0.424 |
| leave-one-day-out | svm_rbf | **0.600** | 0.448 |
| leave-one-session-out (pooled, 15 folds) | knn | 0.583 | 0.444 |
| leave-one-session-out (pooled, 15 folds) | random_forest | 0.609 | 0.398 |
| leave-one-session-out (pooled, 15 folds) | svm_rbf | 0.639 | 0.437 |

(chance level for a balanced 2-class task is 0.50; shuffled-label numbers land close to that, as they
should, confirming the CV/permutation machinery itself is sound.)

- **Real labels beat the shuffled-label control in every split type and every model** -- so there IS
  some genuine, non-random person-related information in this feature set. This is a real, if weak,
  signal, and rules out pure chance/overfitting artifacts.
- But the size of the gap over chance shrinks steadily as the test gets stricter about what's actually
  novel: cross-session-same-day (best: 0.708, ~21 points over a ~0.50 chance level) > leave-one-
  session-out, which still allows other same-day sessions into training (best: 0.639, ~15 points) >
  leave-one-day-out, the strictest test where the entire test day is unseen (best: 0.600, ~10 points).
  **This ordering is exactly the pattern predicted if a meaningful chunk of what these models learn on a
  given day is session/day-specific drift, not a portable person signature** -- the more of "that day"
  the model gets to see in training, the better it does, independent of how much person-specific signal
  is actually present.
- 0.60 accuracy leave-one-day-out on a 2-class task is well short of anything usable for
  identification/authentication, and is only modestly better than the shuffled-label floor (0.42-0.46).

## Final conclusion

**B. WEAK / UNSTABLE person-specific signal.**

- There IS detectable person-related information: real labels consistently and clearly beat the
  session-shuffled-label control across every split type and model (Part 7/9), and the positive control
  (Part 6) rules out "the pipeline can't detect anything" as an explanation.
- But that information is NOT a stable, portable person signature:
  - Raw-signal/heatmap inspection (Parts 1-2) shows no visible person-specific structure -- the strongest
    visible patterns are hardware artifacts and presence-vs-empty, both person-independent.
  - Global feature-space projections (Part 3) show no clustering by person (or by day) at all.
  - Same-person vs different-person distance separability (Parts 4-5) never clears ROC-AUC 0.55 in any
    time scope, including within a single session.
  - The strongest quantitative result (Part 7-9) is a real-vs-chance accuracy gap that is largest when
    train/test sessions come from the SAME day (0.68-0.71) and shrinks by roughly half under genuine
    cross-day evaluation (0.55-0.60) -- the "most important test" per the original brief. This is the
    signature of a model partly learning session/day-specific conditions rather than the person.
- **Practical takeaway**: standing-only CSI on this hardware, at this scale (15 sessions, 3 days, 2
  known people), does not yet support reliable cross-day person identification. Before investing in a
  complex model: (a) collect more days/sessions per person to see whether the cross-day accuracy gap
  narrows with more data or is a hard ceiling, (b) revisit whether the dominant hardware-artifact
  subcarrier bands should be excluded/masked before feature extraction, since they add noise dimensions
  that carry zero identity information by construction (Part 1-2), and (c) check whether the `walking`
  data (gait dynamics, not analyzed in this pass) carries a stronger, more stable signal than the static
  `standing` case -- gait is a well-established stronger biometric channel and was explicitly deferred
  ("let's start with standing data first").
