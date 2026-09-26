# Channel-6 (2026-09-15/16/17) BiLSTM cross-day diagnosis: why FAR-unauthorized stays high

Chain of experiments run 2026-09-20, each a direct follow-up to the previous one's finding. All use
`taskF_identity_or_nonauth` (anjali/barath/non_auth), amplitude-only `BiLSTMAmplitudeOnly`, channel-6
sessions only. Full per-fold numbers are in the referenced CSV logs; this doc is the synthesis.

## 1. EDA (`ml/evaluation/cross_day_eda_ch6.py` -> `cross_day_ch6_eda_findings.md`)

Channel/bandwidth/PHY/MAC routing are all clean and consistent across the three days for the
channel-6 subset. The one real red flag: **2026-09-17's amplitude distribution is measurably the
most different of the three days** (relative mean shift ~14-20% vs 17, vs ~5-9% between 15 and 16),
and 2026-09-15 alone has a packet-rate/label confound (authorized sessions averaged 188Hz vs
unauthorized at 244Hz that day specifically). No AP-placement/environment metadata exists at all
(every session's `notes` field is empty) -- if physical placement changed, there's no way to detect
it directly from data, only via this amplitude-drift symptom.

## 2. Does the paper's exact preprocessing help? (`ablation_ch6_preprocessing_log.csv`, experiments
   E1_raw vs E9_paper_hampel_butterworth)

**No**, on 3-way leave-one-day-out (15/16/17 pooled): AUROC 0.751 (raw) vs 0.747 (paper's
Hampel+Butterworth) -- indistinguishable, well inside seed-to-seed noise (std 0.03-0.12 depending on
fold). Day 17 is the worst fold either way (AUROC ~0.63-0.64 vs ~0.73-0.88 for the other two days),
and denoising makes day-17 specifically *worse* in absolute terms (accuracy 30.6%->17.7%, FAR-none
67%->92%), though still within its own noise band. This matches WhoFi's own finding (RESEARCH_NOTES.md
section 6) that Hampel filtering can hurt identity-discrimination tasks even though it usually helps
activity recognition.

## 3. Excluding day 17 (`ablation_ch6_exclude_day17_log.csv`)

Restricting to just 2026-09-15<->2026-09-16 (both directions) recovers **AUROC 0.83** for both raw and
paper-preprocessed -- a large, real jump from the 3-day-pooled 0.75. Day 17 was the dominant source of
the cross-day gap, not the preprocessing recipe.

## 4. Does threshold tuning fix the still-high FAR-unauthorized? (`compute_threshold_metrics` in
   `run_ablation_ch6_exclude_day17.py`)

**No, not meaningfully.** At the EER operating point (vs. plain argmax): FAR-none drops a lot (down to
2-6%), but **FAR-unauthorized barely moves (64-71% either way)**. The overall EER (~19-28%) looks good
only because "none" (empty room) is trivially easy to separate and dominates the blended metric --
exactly the blended-accuracy trap `ml/reports/day1_findings.md` already warned about. Conclusion: this
is a representation problem (authorized and stranger CSI aren't well separated in feature space), not
a decision-rule problem -- no threshold fixes a boundary that isn't there.

## 5. Is it worse for a genuinely unseen stranger? (`ablation_ch6_leave_one_stranger_out_log.csv`)

**Yes, substantially.** Leave-one-unauthorized-person-out (train on the other 6 strangers + both days'
authorized/none, test on one stranger never seen in training at all) gives **mean FAR-unauthorized@EER
= 84.9% (raw) / 85.2% (paper-preprocessed)** across the 7 distinct strangers in the 15/16 subset --
worse than the 64-71% seen when the stranger was merely from a different day. Per-person FAR is stable
across both preprocessing variants (e.g. abdul 95.0%/96.2%, sumanth 67.0%/64.3%), meaning it reflects
real per-person CSI similarity to anjali/barath, not noise. AUROC looks deceptively high here
(0.92-0.97) purely because "none" inflates the blended metric (FAR-none@EER stays ~0.3% throughout) --
the auth-vs-stranger separation specifically is worse than the headline AUROC suggests.

## Bottom line

Across five independent checks, preprocessing choice (raw vs. the exact arXiv:2507.12854
Hampel+Butterworth recipe) never made a meaningful difference to cross-day generalization. The two
real, load-bearing findings are:
1. **2026-09-17 is a genuine outlier day** (largest measured amplitude drift) and should probably be
   collected again / investigated physically (AP placement, room state) rather than treated as a
   preprocessing problem.
2. **The authorized/unauthorized boundary does not generalize to novel identities** -- this is the
   closed-set-classifier failure mode ARGUS warns about (RESEARCH_NOTES.md section 5/11), not a
   threshold or denoising issue. Fixing it needs either substantially more stranger-identity diversity
   in training, or a genuine open-set/embedding-verification architecture (the project's existing
   `signature_evm` attempt at this is currently stuck at/below chance and would need separate
   debugging before it's usable).

## New scripts added this session

- `ml/evaluation/cross_day_eda_ch6.py`
- `ml/training/run_ablation_ch6_exclude_day17.py` (also added `compute_threshold_metrics`/EER scoring)
- `ml/training/run_ablation_ch6_leave_one_stranger_out.py`
- `ml/data_pipeline/ablation_preprocessing.py`: added `E9_paper_hampel_butterworth` /
  `E10_paper_hampel_butterworth_norm` configs (paper-exact Hampel+Butterworth, matching this repo's
  existing `bilstm_ch6_pipeline.py` recipe, now runnable through the ablation harness's taskF/FAR
  metrics for direct comparison against E1-E8).
