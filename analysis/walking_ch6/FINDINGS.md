# Person-identifiability from walking CSI, channel-6 data -- findings

Same methodology as `analysis/standing_ch6/FINDINGS.md` (same pipeline, features, splits, models),
motion filter swapped to `walking`. Read that report first -- this one only calls out where walking
diverges from the standing result.

## Scope

- **Motion**: `walking` only.
- **Channel/hardware**: channel-6/20MHz sessions only, one fixed receiver board (`ac:27:6e:a5:5b:c8`,
  confirmed present on every collection date). **Correction from the first pass**: channel/bandwidth is
  now detected PER SESSION from the actual packet data (`channel_width_summary`), not assumed from
  `data/README.md`'s "last 3 days" note. That note undercounted -- channel 6 actually spans **6 dates**
  (`2026-09-15` through `2026-09-24`), not 3; `2026-09-15` itself is a mix of channel 6 and channel 11
  sessions, so it's filtered session-by-session, not by date. See `load_data.py::detect_channel6_sessions`.
- **People**: `anjali` (22 sessions) and `barath` (22 sessions) across all 6 channel-6 days -- roughly
  double the session count of the first (3-day) pass.
- Same handcrafted features (mean/std/skew/kurtosis per subcarrier x amplitude/phase + RSSI), same
  200-packet/50%-overlap windows, same session/day-disjoint splits, same session-level label-shuffle
  control. Compute budget trimmed for the larger session count (SVM `max_iter=3000`, smaller
  per-session window cap, fewer permutation repeats) -- see `baselines.py` comments.

## 1-2. Raw signal & heatmaps

- Same hardware-artifact striping as standing (`figures/heatmap_identity_comparison.png`) -- present
  identically in occupied AND empty-room recordings, confirming (again) it's not a person signature.
- **Difference from standing**: the raw amplitude traces (`figures/raw_compare_identities.png`) show
  much more low-frequency envelope structure -- multi-hundred-packet undulations consistent with gait
  periodicity (footsteps modulating the multipath). The residual heatmaps
  (`figures/heatmap_identity_comparison_residual.png`) show visible large-scale blocky bands in the
  calmer subcarrier ranges, unlike standing's speckled residuals -- walking clearly perturbs the
  channel more, and with structure, not just noise. (Some blockiness also appears in a couple of
  empty-room residuals, so not everything blocky here is gait -- other environmental/traffic bursts are
  also possible; this is a visual observation, not by itself evidence of a *person* signature.)

## 3. Feature-space separability

- `figures/feature_space_by_identity.png`, `figures/feature_space_by_day.png`: with 6 days spanning
  more real elapsed time, a **strong two-cluster split appears in both PCA and UMAP -- but it's by DAY,
  not by person.** One cluster is `2026-09-15/16/17`, the other is `2026-09-21/22/24`; empty-room
  windows fall into the same two clusters as occupied ones, confirming this is an environment/hardware
  drift effect between the two collection blocks, not a presence or person effect. **Within either
  day-cluster, Anjali/Barath/stranger/empty-room are still all mixed together** -- no visible
  person-clustering at this coarse level, consistent with the 3-day pass.
- This day-cluster split is exactly what Parts 4-5's same-day-vs-cross-day distance gap (below)
  independently shows: a large day-level shift, symmetric across same-/different-person pairs. It makes
  leave-one-day-out a considerably harder, more meaningful test here than in the original 3-day pass.

## 4-5. Distance-based similarity

| scope | same-person mean dist | different-person mean dist | ROC-AUC |
|---|---|---|---|
| within-session | 35.54 | 36.29 | 0.528 |
| cross-session, same day | 36.34 | 36.29 | 0.503 |
| cross-day | 41.67 | 41.74 | 0.502 |
| overall | 40.68 | 40.83 | 0.505 |

Person separability is essentially nil in every scope, same conclusion as standing. But this table also
makes the day-cluster split from Part 3 numerically explicit: **same-day distances sit around
35.5-36.3, cross-day distances jump to ~41.7 -- a much bigger gap than the person effect, and it hits
same-person and different-person pairs equally** (41.67 vs 41.74 -- no meaningful difference). That's a
real, large day-to-day drift, but it doesn't bias the AUC comparisons above (each row already compares
same- vs different-person distances within one scope) -- it just confirms day is the dominant nuisance
factor, exactly as Part 3 showed visually.

**Key methodological lesson from comparing the two motions**: distance/visualization-based checks stay
weak for walking even though the supervised classifiers below (Part 7-9) find a real, moderately
strong, largely day-stable signal. A simple Euclidean distance over undifferentiated features can be
swamped by uninformative dimensions (here, the day-drift ones) that a trained classifier learns to
down-weight. Absence of separation in Parts 3-5 does not by itself rule out identifiability -- the
cross-validated supervised test is the more sensitive/reliable arbiter.

## 6. Environment / positive control

| split | model | accuracy |
|---|---|---|
| cross-session 5-fold | svm_rbf | 0.957 |
| cross-session 5-fold | random_forest | 0.948 |
| leave-one-day-out (6 folds) | svm_rbf | 0.893 |
| leave-one-day-out | random_forest | 0.813 |
| leave-one-day-out | knn | 0.789 |

Presence detection stays strong even across the harder 6-day leave-one-day-out split (svm 0.893, only
modestly below the 3-day pass's 0.915) -- the pipeline clearly still works under the same strict splits
used below, despite the day-to-day drift documented in Part 3.

## 7-9. Quantitative identity test + permutation control + leave-one-session/day-out

anjali-vs-barath, real vs session-shuffled-label control (`cache/baseline_results.csv`):

**First pass (3 channel-6 dates, 21 sessions):**

| split | model | **real** accuracy | **shuffled-label** accuracy |
|---|---|---|---|
| cross-session (5-fold) | random_forest | 0.712 | 0.453 |
| leave-one-day-out (3 folds) | random_forest | 0.721 | 0.413 |
| leave-one-session-out (21 folds) | random_forest | 0.701 | 0.413 |

**Corrected pass (6 channel-6 dates, 44 sessions -- this is the one that matters):**

| split | model | **real** accuracy | **shuffled-label** accuracy |
|---|---|---|---|
| cross-session (5-fold, 3 usable folds) | knn | 0.586 | 0.473 |
| cross-session | random_forest | **0.710** | 0.436 |
| cross-session | svm_rbf | 0.684 | 0.443 |
| leave-one-day-out (6 folds) | knn | 0.593 | 0.479 |
| leave-one-day-out | random_forest | 0.669 | 0.460 |
| leave-one-day-out | svm_rbf | **0.681** | 0.464 |
| leave-one-session-out (pooled, 44 folds) | knn | 0.606 | 0.482 |
| leave-one-session-out | random_forest | **0.741** | 0.438 |
| leave-one-session-out | svm_rbf | 0.716 | 0.436 |

**This is the headline result, and it answers the exact question that was asked ("was the 3-day plateau
a coincidence?"): no, not really, but there is a small, real cost to the harder test.**

- For standing data (see that report), real-label accuracy dropped sharply as the split got stricter
  (cross-session 0.71 -> leave-one-day-out 0.60) -- the classic sign of the model leaning on
  session/day-specific drift.
- For walking, real accuracy across cross-session / leave-one-day-out / leave-one-session-out is
  **0.68-0.71 / 0.67-0.68 / 0.72-0.74** -- still clustered together, with leave-one-day-out only
  ~2-4 points below cross-session-same-day, not the standing-style collapse. The shuffled-label floor
  stays pinned at 0.44-0.48 throughout, confirming the real numbers are a genuine effect.
- Going from 3 to 6 channel-6 days (with a much larger day-to-day drift visible in Part 3's PCA/UMAP
  and Part 4-5's distance jump) did shave leave-one-day-out accuracy from 0.72 down to 0.68 -- a real,
  modest generalization cost as the test gets harder and more heterogeneous, but nowhere near the
  ~20-point standing-style collapse. leave-one-session-out, if anything, improved (0.70 -> 0.74),
  consistent with there simply being more training data per fold now (44 sessions vs 21).
- **Bottom line**: the person signature in walking data survives a considerably harder, more
  realistic cross-day test (6 days spanning more elapsed time and a documented environment shift) with
  only a modest accuracy cost. It was not a 3-day coincidence.

## Final conclusion

**Leaning A -- real, mostly day-stable person-specific signal, of moderate strength, that holds up
(with a small accuracy cost) under a considerably harder 6-day test than the original 3-day pass.**

- Real labels beat the shuffled-label control by a large, consistent margin (roughly 20-30 points) in
  every split type, across both the 3-day and 6-day passes, including the strictest one
  (leave-one-day-out) -- this rules out chance.
- Critically, unlike standing, **the accuracy does not collapse under the stricter cross-day test** --
  leave-one-day-out (0.68) sits close to cross-session-same-day (0.68-0.71), not dramatically below it.
  That is the specific pattern the original brief asked leave-one-day-out to check for, and it holds
  even after expanding from 3 to 6 days and correcting the channel-6 date range.
- Expanding from 3 to 6 days did cost something: leave-one-day-out real accuracy went from 0.72 (3 days)
  to 0.68 (6 days) as the test got harder (bigger day-to-day drift, documented in Parts 3-5). So the
  3-day plateau was slightly optimistic, but the qualitative conclusion -- day-stable, not
  session/day-confounded -- survives.
- Caveats against calling this "strong" outright:
  - Only 2 known identities. The day-stability is now tested over 6 real collection days, which is much
    more convincing than 3, but the open-set stranger-rejection problem is still untested.
  - Peak accuracy is ~0.68-0.74 on a 2-class task -- clearly informative, far from chance, but well
    short of what a practical identification/authentication system needs.
  - The simple distance/visualization checks (Parts 3-5) show nothing for PERSON -- they do, however,
    clearly reveal a large DAY-level drift between the two collection blocks (`09-15/16/17` vs
    `09-21/22/24`) that dwarfs the person effect in raw distance/2D-projection terms. The person signal
    only shows up under a trained, cross-validated classifier, which is legitimate but means it's a
    subtler multivariate effect than the day drift is.
- **Contrast with standing**: same hardware, same people, same feature pipeline -- the only difference
  is motion. Standing gave a shrinking, session/day-confounded signal (conclusion B). Walking gives a
  largely day-stable signal of moderate strength that survives a harder, more realistic 6-day test.
  This is consistent with gait being a materially stronger and more person-specific biometric channel
  than static standing, as the standing report anticipated.
- **Recommended next step before any complex model**: the day-drift effect found in Parts 3-5 is worth
  understanding on its own (hardware settling, AP/environment changes between 09-17 and 09-21?) since it's
  the dominant source of variance in this feature space and any production model will have to be robust
  to it. On the person-ID side, this motion/feature combination is a reasonable foundation to build on,
  but expect a modest, real accuracy cost from cross-day generalization, not a free lunch.
