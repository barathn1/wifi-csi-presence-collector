# CSI POC Research Notes — authorized-presence detection

## Context

The data collection pipeline in this repo is complete and verified on real hardware: ESP32-S3 streams
CSI over TCP with ~0% loss, `collector/cli_collect.py` writes labeled sessions to
`data/<label>/<date>/<session_id>/{metadata.json, samples.npz}`, and `dataset.labels` in
`config/config.yaml` already includes exactly `["authorized", "unauthorized", "none"]`.

Goal: a POC where **2 specific people are "authorized"** and the model must flag **anything else**
(a different person, or nobody matching an authorized signature) as unauthorized. This document
consolidates research (the WhoFi paper + 4 parallel literature/web research passes covering data
volume, ESP32/subcarrier viability, standing-vs-walking protocol, preprocessing/postprocessing, and
cross-day generalization) into one reference. **Advisory only — no training-pipeline code has been
written yet.**

---

## 1. WhoFi (arXiv:2507.12869) — how they trained, tested, and what they got

- **Task**: person **re-identification/verification**, not fixed-class classification. They train an
  encoder that maps a CSI sequence to an embedding ("signature") using a contrastive/in-batch-negative
  loss, so same-person embeddings land close together and different-person embeddings land far apart.
  Identity is decided at inference by comparing a new embedding's similarity to enrolled reference
  embeddings — not a softmax over known classes.
- **Model used**: compared LSTM, Bi-LSTM, and Transformer encoders + a linear "signature module" +
  L2 normalization. **Transformer won** (1 layer only — a 3-layer version caused "optimization
  instability" and was worse).
- **Training/test data**: NTU-Fi dataset, 14 subjects × 60 samples (3 clothing conditions) = 840 total,
  **546 train / 294 test**, 3-fold cross-validation with an 80/20 train/val split within that. Each
  sample: 2000 packets × 114 subcarriers × 3 RX antennas (TP-Link N750 routers, Atheros CSI Tool —
  not ESP32).
- **Accuracy reported**:
  - Transformer: **Rank-1 95.5% ± 0.013, Rank-3 98.1%, Rank-5 99.1%, mAP 88.4% ± 0.012**
  - Bi-LSTM: Rank-1 84.5%, mAP 61.2%
  - LSTM: Rank-1 77.7%, mAP 56.8%
- **Preprocessing**: amplitude extraction (`|H| = sqrt(real² + imag²)`), Hampel filter (window=5,
  threshold=3) for outlier removal, linear phase sanitization (removes hardware sync offsets).
  **Finding that contradicts most of the literature**: Hampel filtering *hurt* their accuracy — models
  trained without it did better, likely because re-identification needs the fine per-person
  idiosyncratic variation that generic denoising removes (activity-recognition tasks want the opposite:
  suppress fine noise to see gross motion).
- **Data augmentation** (90% probability during training): Gaussian noise (σ=0.02), amplitude scaling
  [0.9, 1.1], time shift ±5 packets. Explicitly load-bearing for making the small-data Transformer work.
- **Overfitting mitigations**: dropout between layers, single-layer architecture (deeper was worse),
  in-batch negative loss (uses all batch negatives, not just paired/triplet), L2-normalized embeddings.
- **What they don't test**: no cross-day/cross-session/cross-environment evaluation anywhere in the
  paper. The 95.5% should be read as an in-domain number, not evidence of real-world drift robustness.

## 2. Is single-antenna ESP32-S3 CSI (64–128 subcarriers) viable for this task?

Two real ESP32-specific papers point in different directions depending on *what* you ask of the
hardware:

- **Single/known-person identification (your actual task)** — [arXiv:2507.12854](https://arxiv.org/html/2507.12854),
  "Transformer-Based Person Identification via Wi-Fi CSI Amplitude and Phase Perturbations": **ESP32**,
  6 subjects, **99.82% accuracy**, standing still across multiple orientations. This is the closest
  real analog to your setup (few known identities, one person at a time) — and it succeeds with a
  Transformer, on your exact chip family. **Confirmed via full-text read (not just the abstract):**
  exact hardware was **two ESP32 boards (TX + RX), 3m apart, laptop-recorded** — essentially the same
  topology as this repo's default 2-board setup. 100 Hz sample rate, 52 subcarriers, subject stood at
  the midpoint (1.5m from each board) through 6 fixed orientations (0°/45°/135°/180°/225°/315°), 150s
  each = 90,000 samples/subject; an empty-room session was also recorded for comparison (same idea as
  this repo's `none` label). Preprocessing: temporal mean-reduction (halves resolution), Hampel filter
  (window=15, β=3) + exponential smoothing (α=0.8) for amplitude outliers, 5th-order Butterworth
  low-pass (cutoff 10Hz), and linear-regression phase-drift removal using only the first/last
  subcarrier's phase (computationally cheap CFO/SFO correction). Model: dual-branch 1-layer Transformer
  (amplitude and phase encoded separately, mean-pooled, concatenated, linear head), d_model=32, 4 heads,
  d_ff=64, dropout=0.2, 1s sliding windows (100 packets) with 50% overlap. **Full results table**:
  Transformer 99.82% acc/99.81 F1, Transformer+CNN hybrid 99.15%, CNN-only 98.72%, MLP-only (no temporal
  modeling at all) 94.36% — even the weakest baseline hit 94%, meaning for only 6 people the sanitized
  signal is highly separable regardless of architecture. **Important caveat**: the 70/10/20 train/val/test
  split was sequential *within each single continuous 150s-per-orientation recording* — same-session,
  not even cross-session, let alone cross-day (see section 8 and section 11 below for why this matters).
- **Simultaneous multi-person identification (not your task)** — [arXiv:2601.02177](https://arxiv.org/abs/2601.02177),
  "Why Commodity WiFi Sensors Fail at Multi-Person Gait Identification: A Systematic Analysis Using
  ESP32": best method only **56% accuracy**, 97–99% feature-space overlap between subjects. Explicitly
  attributes this to single-antenna/subcarrier spatial-diversity limits being the bottleneck, not the
  algorithm.
- Also relevant: [arXiv:2507.17623](https://arxiv.org/pdf/2507.17623) (SA-WiSense), single-antenna
  ESP32 respiration detection, 91.2% up to 8m using a cross-subcarrier ratio trick to work around
  single-antenna blind spots — evidence that single-antenna limitations are workable-around for
  presence-style tasks with the right feature engineering.
- Espressif's own `esp-csi` / `esp_wifi_sensing` (github.com/espressif/esp-csi) targets presence/motion
  detection specifically, not fine-grained identity — consistent with the pattern above.
- **New (section 11): amplitude carries almost all the identity signal on single-antenna/single-radio
  hardware.** ARGUS's channel ablation on a single-antenna Raspberry Pi CSI probe found amplitude-only
  reaches 79.7% (statistically indistinguishable from the full amplitude+phase+phase-delta model at
  78.9%), while phase-only drops to 41.5% and phase-delta-only to 3.0%. Some support for not
  over-investing engineering time in phase sanitization if you need to simplify — though this is
  setup-dependent: on ARGUS's separate multi-antenna WiMANS benchmark, calibrated phase helped *more*,
  especially on noisier 2.4GHz configs. Don't treat this as "phase is useless," just "amplitude is the
  safe default to get working first." **A third paper (Trans-MAML, section 11) independently made the
  same choice for the same stated reason** — amplitude is more robust to hardware/sync imperfections
  than phase — reinforcing amplitude-first as the sensible default rather than a one-off finding.
- **New (section 11)**: Trans-MAML's hardware limitation section states they only validated **1x3 and
  3x3** antenna configurations — this repo's ESP32-S3 (effectively 1x1 on both ends) is below the
  lowest antenna count either ARGUS or Trans-MAML actually tested. Neither paper proves single-antenna-
  to-single-antenna CSI works as well as their multi-antenna-receiver results suggest; treat their
  numbers as an encouraging upper-bound signal, not a guarantee for this exact hardware.

**Conclusion**: your hardware is viable for your actual task (verify 1 of 2 known people, one at a
time) — you are not attempting the task that was shown to fail (simultaneous multi-person ID).

**ESP32-specific quirks** (already handled in this project's firmware):
- `first_word_invalid`: documented hardware bug zeroing the first 4 bytes of CSI on some frames
  (espressif/esp-csi issues #4, #45, #146) — this project's wire format already carries a
  `first_word_invalid` flag bit per sample for exactly this reason.
- Promiscuous-mode capture rate/density varies with ambient traffic — matches what this project's own
  capacity testing already found (~60-85 Hz observed, ambient-traffic-driven).

## 3. Standing or walking?

Mixed across papers, and it matters which one is the closer analog to your hardware:
- NTU-Fi/WhoFi (multi-antenna router): short **walking** pass — gait/Doppler-driven signature.
- The ESP32-specific 99.82% paper (arXiv:2507.12854): **standing still, multiple orientations** —
  body-shape-induced amplitude/phase distortion, no gait needed.

Given single-antenna ESP32 has weaker Doppler/motion resolution (per the 56%-accuracy multi-person
finding above), **weight collection toward standing at a few different orientations/positions** —
this matches the paper that actually succeeded on ESP32-class hardware. Add a handful of walking
passes as a secondary variant for robustness, not as the primary protocol.

**Update (section 12)**: a dedicated follow-up search found **zero** papers demonstrating single-antenna
commodity WiFi CSI walking/gait identification working well for one person at a time — every
high-accuracy walking-gait result uses multi-antenna hardware, and the only single-antenna
high-accuracy walking result uses 60GHz mmWave radar (a different sensing modality entirely, not
WiFi). This upgrades the recommendation above from "weaker, so prefer standing" to **"walking on this
exact hardware class is genuinely unvalidated by any known paper — treat it as an open experiment, not
a fallback with a known answer."**

## 4. How much training data is actually needed?

No published rule-of-thumb exists; real numbers span roughly two orders of magnitude by task
difficulty:
- NTU-Fi (WhoFi's dataset): 14 subjects × 60 short samples = 840 total.
- NTU-Fi HAR (6 activities): 156 train + 44 val samples *per class*.
- SignFi (276 gesture classes): only 10–20 samples/class, fed into a 9-layer CNN, reported sufficient.
- Widar3.0 (cross-domain generalization, a much harder goal): 258,000 instances / 8,620 minutes — a
  ceiling for cross-environment generalization, not a minimum for a basic POC.
- A companion transformer person-ID paper from WhoFi's lab used only 9 subjects.
- **New (section 11)**: ARGUS's EHealth dataset is a genuine large-cohort counterpoint — 154 subjects,
  17 activities × 60s each. At that cohort size, same-session Top-1 accuracy is only **78.88% (6s
  window) to 84.85% (60s aggregated)** — much lower than the 99.82% for 6 subjects in section 2, on a
  comparably "static/passive" task. **Cohort size is a first-order driver of achievable accuracy,
  independent of architecture.** Your task (2 authorized identities to recognize, everything else
  rejected) sits far closer to the easy 6-subject regime than the hard 154-subject one — a reason for
  cautious optimism, not a reason to expect 99%+ automatically.

No paper states a quantified "transformers need Nx more data than CNN/LSTM" rule for CSI — that's
absent from the literature, not just unfound. The one controlled comparison (WhoFi's own 1-layer vs.
3-layer ablation) supports "keep it shallow," not "avoid transformers entirely."

## 5. Model choice — revised conclusion

**A small (1-layer) Transformer is a reasonable primary attempt, not just a stretch goal** — this is
now directly supported by ESP32-specific evidence (arXiv:2507.12854, 99.82%), not just WhoFi's
multi-antenna result. Keep LSTM/Bi-LSTM as a fallback if training proves unstable (matches WhoFi's
own finding that going deeper, not staying with attention itself, was what caused instability).

Frame it as **embedding/verification** (contrastive or triplet loss, cosine/L2 similarity + threshold
at inference), not a fixed 3-way classifier — a classifier can only recognize the specific unauthorized
people it was trained on; verification-by-similarity generalizes to a genuinely novel intruder, which
is what "flag anything other than these 2 people" actually requires.

**New, directly on point (section 11)**: ARGUS — a 2026 paper built specifically to study open-set
rejection (detecting people *not* in the enrolled set) — states plainly that this problem is **"not
solved"** even in their own system, and warns explicitly: *"Passive identification systems such as
Argus should not be used as a covert identity gate; the open-set results show that unknown users can
still be assigned confident closed-set identities without a separate verification layer."* Their
unknown-user rejection AUROC was only **0.744–0.754** — barely better than a coin flip plus a bit. This
is strong, recent, independent validation that a plain closed-set classifier is the wrong tool here;
the embedding/verification + tuned-threshold approach above is what actually addresses "flag anything
other than these 2 people," and skipping straight to a classifier is the exact mistake ARGUS is warning
against.

## 6. Preprocessing pipeline (standard order across papers)

Raw complex CSI → amplitude (`sqrt(imag²+real²)`) and phase extraction → outlier/noise filtering
(Hampel filter and/or Butterworth low-pass are most commonly cited) → phase sanitization/unwrapping
(linear-fit removal of per-packet sync offsets — treated as near-mandatory, raw phase is otherwise
unusable) → per-subcarrier normalization (z-score or min-max) → fixed-length windowing → optional
STFT/spectrogram or PCA.

- **Test Hampel filtering both ways on your own data** rather than assuming it helps — WhoFi found it
  hurt for re-identification specifically, which is unusual vs. activity-recognition literature where
  it's near-universally beneficial. Task-dependent, not settled.
- This project's raw `(imag, real)` int8 pairs per subcarrier (see `collector/inspect_npz.py` output)
  are exactly the right starting point for amplitude/phase extraction.
- CSI length varies per packet (64 vs 128 subcarriers, legacy vs HT frames) — a model needs one
  consistent input shape per window, so either filter to one frame type or pad/mask consistently.
- **New (section 11)**: Trans-MAML's autoencoder-based background subtraction is a different, learned
  alternative to Hampel/Butterworth-style filtering — train a small autoencoder to predict the static
  "empty room" signal from a mix of empty-room and occupied recordings, then subtract that prediction
  from any new recording to isolate the human-motion residual. Their own ablation found it gives only a
  **modest** improvement (mainly reduced variance, not higher mean accuracy — Table 9: 98.1→98.3% at
  1-shot, 99.4→99.9% at 5-shot) in their controlled room, so it's a nice-to-have refinement worth trying
  given this repo already collects `none` (empty-room) sessions for exactly this purpose, not a
  must-have before a first model attempt.

## 7. Postprocessing

- **Live "is an authorized person here right now" signal**: sliding-window majority voting, or
  requiring N consecutive consistent windows before flipping state (debouncing) — used across several
  presence-detection papers (e.g. CRONOS, arXiv:2211.10354; Time-Selective RNN, arXiv:2304.13107, which
  reports >97% with window-based smoothing). Exact window sizes/thresholds are engineering judgment,
  not standardized in the literature.
- **For an embedding/verification model specifically**: cosine similarity (or L2-normalized dot
  product) scoring, with **EER** (equal error rate — where false-accept rate equals false-reject rate)
  as the standard metric for tuning the accept/reject threshold. A 2025 SoK on WiFi CSI biometrics
  (arXiv:2511.11381) recommends per-class EER and warns the field inconsistently reports thresholds,
  and states EER should not exceed 5% for security-grade deployments.
- **New (section 11)**: ARGUS confirms segment-level aggregation is a real, free accuracy gain, not
  just a presence-detection trick — averaging softmax logits over 19 overlapping 6s windows across a
  60s segment raised their Top-1 from 78.88% to 84.85% with **zero retraining**, same trained model.
  Same mechanism as the sliding-window majority voting above, just soft logit-averaging instead of a
  hard vote — evidence this class of technique generalizes across both presence-detection and
  person-ID tasks.

## 8. Cross-day / domain generalization — real numbers, and it's mostly unsolved

WhoFi itself has zero cross-day evaluation. From dedicated research into this specific problem:

- **Magnitude of the gap**: one study correcting for subject-overlap leakage between train/test splits
  (the same failure mode as same-day leakage) found accuracy dropped **27–32 percentage points**
  (e.g. a 2D-CNN went from 92.2% → 66.4%). Same-session numbers — including WhoFi's headline 95.5% —
  should be assumed optimistic relative to real cross-day deployment until proven otherwise on your
  own held-out day.
- **GaitID** (WASA 2020) found raw CSI amplitude "takes drastically different shapes on different days"
  for the *same static link*, but **motion/velocity-derived features stayed discriminative across
  days**. Actionable signal: prefer motion/delta-based features over raw absolute amplitude if cross-day
  robustness matters more than peak same-day accuracy.
- **Techniques used elsewhere in WiFi sensing** (not yet validated for person-verification specifically):
  adversarial domain adaptation (EI, MobiCom 2018), CSI-ratio/conjugate multiplication to cancel shared
  hardware phase noise between antenna pairs, Doppler-based domain-independent features (Widar3.0's
  BVP, 82.6–92.4% across unseen locations/environments with zero retraining), transfer learning
  (CrossSense, MobiCom 2018: ~20% → 80-90%+ cross-site after adaptation), few-shot recalibration
  (FewSense: 5 labeled samples/class per new domain → 82.7–96.5%).
- **No published re-enrollment-interval guidance exists for CSI verification** — this is a genuine gap
  in the literature, not something to borrow a number for. This project's own held-out Day 3 test
  (see collection schedule below) is the only way to get an honest, real number for your setup.
- **New (section 11)**: ARGUS is a second, independent, more recent (2026) confirmation that
  cross-environment generalization is weak, this time with exact numbers on a public multi-room
  benchmark (WiMANS): zero-shot transfer to a held-out room reached only **14–28% exact-match**
  (vs. ~95% in-room). Counter-intuitively, **mixing a small amount of target-room data into training
  beat the standard "pretrain-then-fine-tune" recipe at every data budget tested** (1–50% of target
  data) — contrary to common transfer-learning practice. Worth remembering if this POC is ever
  adapted to a new room: don't assume fine-tuning a pretrained model is the best move, try mixed
  training first.
- **New (section 11)**: Trans-MAML provides a fourth independent data point on the same theme, from a
  completely different task (activity recognition, not person-ID) and research group: moving to a new
  room dropped their true open-set-like "novelty detection" accuracy from 98.9% (same room) to
  **73.3%** (new room) — a ~25-point drop, the same order of magnitude as the 27-32 point cross-day
  drop already cited above. Cross-environment/cross-day generalization being the hard, unsolved part
  of this space is now a consistent pattern across at least four independent papers/tasks, not an
  artifact of any one dataset or method.

## 9. Recommended multi-day collection schedule

Multi-day collection is feasible for this POC, and 2+ distinct unauthorized people are available
(including one that can be held out entirely for Day 3) — the strongest version of the generalization
test is achievable, not just a fallback.

| Day | authorized/person1 | authorized/person2 | unauthorized/personX | none (empty room) |
|---|---|---|---|---|
| Day 1 | 2-3 sessions | 2-3 sessions | 2-3 sessions | 1-2 sessions |
| Day 2 (different time of day; restart the hotspot first) | 2-3 sessions | 2-3 sessions | 2-3 sessions | 1-2 sessions |
| Day 3+ (held out — never used in training) | 1-2 sessions | 1-2 sessions | 1-2 sessions (**ideally a genuinely new/different unauthorized person**) | 1 session |

Weight sessions toward **standing at varied positions/orientations** (section 3), with a few walking
passes as secondary. Day 1+2 = training pool; Day 3 = held out entirely for an honest cross-day number,
not a pooled random split (section 8).

## 10. How to proceed

1. Collect per the schedule above using the existing, unchanged `collector/cli_collect.py` — no
   pipeline changes needed, `config.yaml`'s label set already covers this.
2. When ready to build the training pipeline: windowing (fixed-length sequences from the per-packet
   `samples.npz` rows), amplitude+phase extraction, phase sanitization, normalization, a small (1-layer)
   Transformer or LSTM/Bi-LSTM encoder trained with contrastive/triplet loss, a batch sampler that
   guarantees identity diversity per batch, heavy data augmentation, and an enrollment + EER-tuned
   threshold evaluation step.
3. Report the Day-3 held-out accuracy as the real POC number, not a pooled/random split.

## 11. Update (2026-09-08): two new papers reviewed

### ARGUS (arXiv:2608.14670) — full text read

**"ARGUS: Attention-Guided Transformers for Scalable Person Identification Using Wi-Fi Telemetry,"**
Bhatia, Kocheta, Li, Obraczka (UC Santa Cruz), Aug 2026. The most directly relevant new paper found —
large-cohort, open-set-aware, and explicitly warns against exactly the failure mode this POC must avoid.
Referenced inline above (sections 2, 4, 5, 7, 8); full detail here.

- **Task**: passive person identification from CSI, closed-set *and* open-set (rejecting unenrolled
  people) — the second half is the direct analog of "flag anyone but these 2 people."
- **Hardware/datasets**: two datasets, neither ESP32:
  - **EHealth** (154 participants, Galdino et al. 2023): single-antenna Raspberry Pi 4B running NEXMON
    + laptop client, 5GHz channel 36, 80MHz BW, 256 subcarriers (234 usable), 3m×4m room. 17
    standardized 60s positions/activities per subject (14 static) — all electronics/phones removed from
    the room to avoid EM interference.
  - **WiMANS** (public multi-user benchmark, Huang et al. 2024): 11,286 three-second recordings across
    classroom/meeting-room/empty-room, 2.4GHz and 5GHz, 0-5 of 6 enrolled users per recording.
- **Method**: raw CSI → **"statgram"** (a compact statistical map: per-window robust-normalized
  amplitude/phase/phase-delta, reduced to statistic rows — mean, std, percentiles, energy — over
  grouped subcarrier bins) → coarse patches → **decoder-only Transformer** (causal attention, CLS token
  placed last) → softmax (closed-set) or independent binary logits (multi-user). Attention-guided
  occlusion then ranks which statgram patches matter, for input compression.
- **Closed-set accuracy (EHealth, 154-way)**: Top-1 **78.88% ± 1.62%** on a single 6s window, rising to
  **84.85% ± 1.31%** aggregating 19 overlapping windows over 60s (Top-3 98.61%, Top-5 99.26%). Beats a
  raw-CSI Transformer baseline (THAT) by 7.75 points at 60s while using 4.4x fewer FLOPs. A raw-CSI
  Transformer *degrades* with more context (77.10% at 60s, down from 82.37% at 6s) while the
  statgram model *improves* with context (78.88% → 84.85%) — long raw temporal context alone doesn't
  help, compact stable representations do.
- **Open-set rejection (EHealth, 31/154 identities withheld)**: known-user accuracy on the other 123
  drops to 78.69%; unknown-user rejection AUROC is only **0.744** (max-softmax confidence) / **0.754**
  (negative entropy). Explicit conclusion: *"open-set rejection is also not solved"* — see the direct
  quote in section 5 above. 61.9% of the remaining Top-1 errors are to *adjacent* participant IDs
  (vs. 1.3% expected by chance) — errors are structured (likely collection-order/session artifacts),
  not random noise.
- **Cross-room transfer (WiMANS, leave-one-environment-out)**: zero-shot 14-28% exact-match on a
  held-out room; mixed training (source rooms + x% target) beats fine-tuning at every budget tested;
  reaches 82-93% with 25% of target-room data (~376 recordings).
- **Channel ablation**: amplitude-only 79.7% ≈ full 3-channel 78.9%; phase-only 41.5%; phase-delta-only
  3.0% — on WiMANS specifically, calibrated phase helps more, especially at 2.4GHz.
- **Efficiency**: 1.92M params, 0.061 GFLOPs/window, 27x fewer FLOPs than the THAT baseline for
  comparable WiMANS accuracy (95.07% vs 95.94% mean exact-match across 9 room/band configs, within 1.23
  points on average).
- **Explicit governance warning** (worth keeping in mind for a real deployment, not just the POC):
  *"Passive biometric sensing also requires explicit governance... should require informed enrolment
  and consent, visible opt-out mechanisms, retention limits... Passive identification systems such as
  Argus should not be used as a covert identity gate."*

### Trans-MAML (IEEE Access vol. 14, pp. 78740-78756, 2026, DOI 10.1109/ACCESS.2026.3696430) — full text read

**"Trans-MAML: Meta-Learning Enhanced Transformers for Human Activity Recognition in WiFi CSI
Sensing,"** Rifdah, Dutta, Matsumaru, He (open-access, CC-BY — the user supplied the PDF after IEEE
Xplore blocked automated fetching). **This is an activity-recognition paper, not person-ID** — it
classifies *what someone is doing* (walk/sit/stand/pick up/lie down/nothing), not *who* they are. Still
relevant here for three reasons: it uses the same amplitude-focused, Transformer-based approach; it
directly tackles "recognize something my model never trained on" (structurally similar to the
authorized/unauthorized open-set problem, just for activities instead of identities); and it introduces
a background-subtraction preprocessing technique this repo's existing `none`-label sessions could feed
directly.

- **Hardware**: TP-Link Archer C7 router (**1 antenna**, transmitter) + a NUC with an Intel 5300 NIC
  (**3 antennas**, receiver), 5GHz, 40MHz bandwidth, 1kHz packet rate, 30 subcarriers. 9m×6m room, walls
  covered in white cloth to reduce multipath clutter, 3 volunteers, 6 activities (nothing/walk/sit
  down/stand up/pick up/lie down), 10s recordings x10 repeats. Compared against the public **WiMANS**
  dataset (3 TX antennas x 3 RX antennas = "9-antenna" configuration, 5 participants, 6GHz/2.4GHz
  variants). **Neither dataset used a true single-antenna-to-single-antenna (1x1) link** — their own
  hardware's *transmitter* is single-antenna but the *receiver* still has 3. The paper's own stated
  hardware limitation: *"We tested only under 1x3 and 3x3 antenna array configurations... this may
  limit its performance across diverse deployment scenarios."* This repo's ESP32-S3 boards are
  effectively 1x1 on both ends — **below the lowest configuration this paper actually validated**, so
  its "98.9% at 3 antennas" result is a useful signal, not a proven floor for our exact hardware.
- **Method**: amplitude-only CSI (explicitly chose amplitude over phase for the same reason as ARGUS —
  "more robust to hardware imperfections and synchronization errors" — a third independent paper
  now favoring amplitude-first). A simplified two-stream Transformer (their own ablation: replacing the
  THAT baseline's multi-scale-CNN sublayer with a plain feed-forward layer cuts params from 4.95M to
  2.43M for a statistically insignificant accuracy change, 64.85% vs 63.33%, paired t-test p=0.589) is
  meta-trained with **first-order MAML** for few-shot adaptation to a held-out activity class (leave-one-
  activity-out, LOAO). A novel **autoencoder-based background-subtraction** preprocessing step learns a
  static "environment" prototype from empty-room + human-action samples and subtracts it, isolating the
  human-motion residual before it reaches the Transformer (Table 8: statistically validated the empty-
  vs-occupied distinction survives in both raw and preprocessed CSI, p<0.001 either way, t-stat improves
  from -12 to 25.4 after preprocessing) — conceptually close to what this repo's `none` (empty-room)
  label already collects for, if a background-subtraction step is ever added to a training pipeline.
- **Headline few-shot adaptation numbers** (Table 5, mean accuracy % ± std over 5 runs, LOAO
  protocol): WiMANS 1/3/5-shot = **99.1±1.7 / 99.4±1.8 / 99.4±2.3**; their own dataset 1/3/5-shot =
  **93.3±14.7 / 96.7±18.0 / 98.9±1.6**. Note the huge variance at 1-shot on their own dataset (±14.7,
  ±18.0) — the mean looks great, but few-shot-from-1-example is genuinely unstable, not just "slightly
  noisy." Every other baseline tested (original THAT, CNN-Meta-Learning, Dual-Path ProtoNet, CNN-LSTM+
  Reptile, BeamSense) either failed outright or scored far lower — **the original (non-meta-learning)
  THAT model gets a structural 0% here, because a plain closed-set classifier always predicts one of
  its known classes and has no mechanism to say "this is a new class I've never seen."** This is now
  the *third* independent paper/task (after ARGUS's person-ID and this repo's own section 5 reasoning)
  converging on the same point: a closed-set classifier structurally cannot do "flag anything I wasn't
  trained on" — you need something built for it (few-shot/meta-learning here, embedding/verification
  for the identity case).
- **Novelty detection (Table 7)** — their closest analog to true open-set rejection (classify a sample
  as "new/unknown" without assigning it to any known class, rather than just fast-learning a new labeled
  class): WiMANS 1/3/5-shot = 95.9±9.3 / 93.1±10.9 / 98.9±2.8; their dataset = 86.3±4.8 / 96.5±1.2 /
  98.9±2.7. Same-environment, these numbers look strong.
- **Cross-environment result (Table 10) — this is the important one for section 8.** Move to a new
  room and the picture changes a lot: adaptation-task 5-shot drops to 96.5±7.9 (WiMANS) but a much
  larger drop to 71.1±29.5 (their own dataset, n-class=3) — and **novelty detection under cross-
  environment** (the true open-set-like case) drops further still: WiMANS falls from 98.9±2.8
  (same-room) to **73.3±0.4 (new room)**, roughly a **25-point drop**, same order of magnitude as the
  27-32 point cross-day drop already cited in section 8 from a completely different paper/task. Their
  own quote: *"detecting completely unseen activity classes represents a stricter open-set
  challenge"* under environment shift — a fourth independent confirmation that cross-environment/
  cross-day generalization is the hard, still-mostly-unsolved part of this whole space, regardless of
  whether the task is activity recognition, person-ID, or (implicitly) authorized-person verification.
- **Explicit limitations acknowledged by the authors** (worth taking at face value): only 6-9 activity
  classes tested, only 10-66 samples per class (their own dataset: **only 10 samples/class** — flagged
  by the authors themselves as reducing statistical robustness); background-subtraction "shows only
  minimal improvement" in their controlled/static room and is untested in "complex environments with
  more moving objects"; only single-unseen-activity-at-a-time was tested, not simultaneous multi-activity
  or **multi-person** scenarios — the latter listed explicitly as future work, consistent with the
  multi-person-is-still-hard theme already in section 2 (arXiv:2601.02177's 56%-accuracy finding).

## 12. Follow-up check (2026-09-09): single-antenna gait, a real cross-day success story, and daily calibration

Triggered by three direct questions: can single-antenna hardware do *walking* identification well
(not just standing)? Has *any* paper solved cross-day CSI person-ID? Does recording a fresh empty-room
baseline each day plausibly help? Answers below, from a real literature search, not just the papers
already in this document.

### Single-antenna + walking + high accuracy: not found

Searched specifically for single-antenna (ESP32-class) commodity WiFi CSI gait/walking identification
for **one person at a time** (not the already-known 56%-accuracy multi-person failure,
arXiv:2601.02177). Every walking-gait paper found that reports strong accuracy uses **multi-antenna**
hardware — almost always the 3-antenna Intel 5300 NIC or multiple receivers:
[Wi-GPD](https://dl.acm.org/doi/10.1145/3746639) (3-antenna, 84.75-99.5%),
[GaitSense](https://dl.acm.org/doi/10.1145/3466638) (1 TX + 6 multi-antenna RX),
[Wi-Gait](https://www.sciencedirect.com/science/article/abs/pii/S1389128623001962) (~92.9%, 10
subjects), [GaitID](https://tns.thss.tsinghua.edu.cn/widar3.0/data/WASA20_GaitID_paper.pdf) (>93.2%),
[WiDIGR](https://ieeexplore.ieee.org/document/8901187/) (78-93%) — all multi-antenna. The one
genuinely single-antenna, single-person, high-accuracy (96.1-98.3%) walking result found,
[GaitCube](https://ieeexplore.ieee.org/document/9440988/), uses **60GHz mmWave radar**, a completely
different sensing modality — not WiFi CSI, not ESP32-class hardware, not replicable with this repo's
setup. **Conclusion: no paper has proven single-antenna commodity WiFi CSI walking identification
works well.** This isn't just "less validated than standing" (section 3) — it's *unvalidated,
period*. Treat walking-based ID on this hardware as a genuine open experiment, not a fallback with
a known answer.

### Cross-day: one real success story exists, and it matches your protocol better than you'd think

**[WiPIN](https://arxiv.org/abs/1810.04106)** (GLOBECOM 2019, Wang/Han/Lin/Ren) is the one paper found
with a genuine day-disjoint test that reports good results: **static body-shadowing identification**
(standing/present, not gait — i.e. the same style of task section 3 already recommends), 10 subjects,
CSI recorded over **15 consecutive real calendar days**.
- **Strategy 1** (train once on day 1, never retrain, test on each later day): accuracy degrades
  *gradually*, not catastrophically — **~98% down to ~90% by day 10**.
- **Strategy 2** (periodically fold each new day's data into training as it arrives): accuracy holds
  at **~98% across all 15 days**.

Two things make this unusually relevant to your specific plan, more than any other paper cited so far:
1. **It's the same task family you're already planning** (static presence, not gait) — unlike the
   room-shift papers elsewhere in this doc (ARGUS, Trans-MAML), which is why its degradation is far
   gentler (8 points over 10 days) than their ~25-32 point cross-*room* drops. Cross-day-in-the-*same*
   room looks meaningfully easier than cross-room — worth internalizing as a distinct, easier problem
   than the scarier room-shift numbers elsewhere in this document.
2. **It directly answers this project's earlier-flagged gap** ("no published re-enrollment-interval
   guidance exists," section 8) — Strategy 2 *is* a concrete re-enrollment strategy: periodically
   re-including recent real data in training keeps accuracy flat instead of decaying. If your Day-3
   held-out result comes back meaningfully worse than Day-1/2, this is the first thing to try before
   concluding the whole approach doesn't work.

**Caution, same pattern as WhoFi**: two papers that *look* multi-day turned out not to test cross-day
generalization at all — [an 8-month authentication study](https://par.nsf.gov/servlets/purl/10292008)
(94-97%) and a [2025 mmWave-WiFi gait paper](https://arxiv.org/abs/2510.08160) spanning 3 real days
(91.2%) both used a **random pooled split** across all collected days rather than holding a day out —
the exact same leakage pattern already flagged for WhoFi in section 1/8. **When you build your own
training script, the train/val/test split must be day-disjoint, not a random shuffle across all
collected days** — a random split would silently produce an inflated, meaningless number, not a real
cross-day result. A 2025 field survey,
[SoK: Security Evaluation of Wi-Fi CSI Biometrics](https://arxiv.org/abs/2511.11381) (already cited in
section 7), independently confirms "drift over time" remains a named, unresolved open challenge for
the field generally — this is a real, acknowledged gap, not something specific to this project.

### Daily empty-room calibration: unproven for this exact use, but one closely-related ESP32 result is genuinely promising

No paper validates "fresh empty-room baseline each day -> measured cross-day accuracy gain" for
person-ID, gait, or activity recognition — that exact claim doesn't exist in the literature searched.
But one very close analogue does, on your exact hardware family:
**[OpenCSI](https://arxiv.org/abs/2607.26665)** ("Self-Calibration Layer for Heterogeneous Mesh
Wireless Sensor Networks") bootstraps a short empty-room baseline **at each deployment**, Z-score-
normalizes every link against its own quiet-period statistics, and reports **F1 up to 0.99 vs. 0.87**
for standard normalization, **zero-shot across 3 different rooms and 3 different ESP32 hardware
generations, with no retraining**. The catch: it's scoped only to **binary occupancy/presence
detection** (empty vs. moving), not gait or person-ID, and its validated transfer axis is
cross-room/cross-hardware, not explicitly cross-day. Still — it's real evidence that "record a short
empty-room baseline, normalize against it" measurably helps *something* CSI-related generalize on
ESP32 hardware specifically, which is more direct support than Trans-MAML's same-room-only autoencoder
result (section 11) had on its own.

**Practical takeaway**: this is a cheap, easy-to-test hypothesis worth trying on your own data, not a
proven fix. You already collect `none` (empty-room) sessions for this exact purpose — when you build
a training pipeline, try feeding the model raw amplitude/phase *and* try a version normalized against
that day's `none` session, and compare. Nothing in the literature guarantees it helps for person-ID
specifically, but nothing contradicts it either, and OpenCSI's cross-room result is a genuinely
encouraging, same-hardware-family sign.

## Sources

- WhoFi: [arXiv:2507.12869](https://arxiv.org/abs/2507.12869)
- ESP32 Transformer person-ID (99.82%): [arXiv:2507.12854](https://arxiv.org/html/2507.12854)
- ESP32 multi-person gait-ID failure analysis (56%): [arXiv:2601.02177](https://arxiv.org/abs/2601.02177)
- SA-WiSense (ESP32 respiration): [arXiv:2507.17623](https://arxiv.org/pdf/2507.17623)
- SenseFi / NTU-Fi HAR benchmark: [arXiv:2207.07859](https://arxiv.org/pdf/2207.07859), [arXiv:2506.11165](https://arxiv.org/pdf/2506.11165)
- Widar3.0: [TPAMI paper](https://tns.thss.tsinghua.edu.cn/widar3.0/data/TPAMI_Widar3.0_paper.pdf)
- GaitID: [WASA 2020](https://tns.thss.tsinghua.edu.cn/widar3.0/data/WASA20_GaitID_paper.pdf)
- CrossSense: MobiCom 2018, [ACM](https://dl.acm.org/doi/10.1145/3241539.3241570)
- EI (adversarial domain adaptation): [MobiCom 2018](https://cse.buffalo.edu/~lusu/papers/MobiCom2018.pdf)
- FewSense: [arXiv:2203.02014](https://arxiv.org/pdf/2203.02014)
- Data-leakage/cross-subject accuracy drop study: [PMC11679234](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679234/)
- Optimal CSI preprocessing survey: [arXiv:2307.12126](https://arxiv.org/pdf/2307.12126)
- SoK: WiFi CSI biometrics security: [arXiv:2511.11381](https://arxiv.org/abs/2511.11381)
- CRONOS: [arXiv:2211.10354](https://arxiv.org/pdf/2211.10354)
- Time-Selective RNN presence detection: [arXiv:2304.13107](https://arxiv.org/pdf/2304.13107)
- Espressif esp-csi: [github.com/espressif/esp-csi](https://github.com/espressif/esp-csi)
- ARGUS (large-cohort + open-set person-ID): [arXiv:2608.14670](https://arxiv.org/abs/2608.14670)
- Trans-MAML (meta-learning transformer HAR, IEEE Access 2026): [DOI 10.1109/ACCESS.2026.3696430](https://doi.org/10.1109/ACCESS.2026.3696430)
- WiPIN (real 15-day cross-day person-ID success): [arXiv:1810.04106](https://arxiv.org/abs/1810.04106)
- OpenCSI (ESP32 empty-room self-calibration, cross-room/cross-hardware): [arXiv:2607.26665](https://arxiv.org/abs/2607.26665)
- GaitCube (single-antenna mmWave, not WiFi CSI -- cited as the closest non-match): [IEEE](https://ieeexplore.ieee.org/document/9440988/)
- DATTA (test-time domain adaptation for cross-domain WiFi HAR): [arXiv:2411.13284](https://arxiv.org/html/2411.13284)
