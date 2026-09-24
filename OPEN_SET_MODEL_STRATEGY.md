# Open-Set Authorized/Unauthorized Detection — Model Strategy

Goal: binary authorized-vs-stranger detection from CSI (not full per-person re-ID),
with open-set rejection of people never seen during training, on a single-link
setup (one TP-Link router as TX, one ESP32/ESP32-S3 as RX/CSI capture).

## Current pipeline config

- Window size: 2s, overlap: 0
- Normalization: zscore (per window)
- Held-out split: room (add a second held-out axis on **day** once multi-day data exists)
- Min frames/window: 10 (short windows dropped, not zero-padded)

## Core signal intuition

The whole approach rests on one assumption: **a given subcarrier index behaves
differently depending on who is walking through the link.** Subcarrier `k`
reacts one way to human 1 and a different way to human 2, because each
subcarrier probes a slightly different frequency/multipath geometry, and body
shape + gait perturbs each of those paths differently. The person-specific
fingerprint lives in the *pattern across subcarriers*, not in any single
subcarrier's average behavior.

This has direct consequences for every choice above:

- Feature engineering (SVM/GBM path) must keep **per-subcarrier** statistics,
  not just pooled/aggregate stats across the whole band — averaging away
  subcarrier identity destroys the signal.
- CNN receptive fields / GNN adjacency should respect subcarrier-index
  structure (neighboring subcarriers are correlated via multipath, but the
  discriminative pattern is the *shape across the full band*, not local
  smoothness).
- Calibration/normalization must be done **per subcarrier**, not as one
  global scalar over the whole window — see the dedicated section below.

## Why this is a harder-than-usual case

- Single TX–RX link → low spatial diversity, no antenna-array tricks to fall back on.
- Open-set + cross-day evaluation combined is rare in the published literature —
  most CSI authentication papers test one or the other, not both, and almost
  none use ESP32-class hardware. Treat this as exploratory territory, not
  "reproduce paper X."
- Published multi-person gait-separation results on commodity ESP32 hardware show
  severe intra-subject vs inter-subject overlap (arXiv 2601.02177). The
  binary authorized/stranger framing is less exposed to this than multi-person
  simultaneous separation, but expect ceiling effects if the authorized group grows large.

## Reference literature

- **CAUTION** (Wang et al., IEEE IoT J. 2022) — few-shot prototypical-network
  embedding + distance-ratio open-set threshold, calibrated using only held-out
  known users as pseudo-strangers (no real intruder data needed). Best conceptual
  template for this project.
- **Wii** (Sensors 2017) — GMM stranger-reject + SVM per-user identity. Classical
  two-stage precedent for the SVM path below.
- **"Why Commodity WiFi Sensors Fail at Multi-Person Gait Identification"**
  (arXiv 2601.02177) — hardware-ceiling caution for ESP32 CSI.
- **SoK: Security Evaluation of Wi-Fi CSI Biometrics** (arXiv 2511.11381) —
  flags cross-day + open-set combined evaluation as an unsolved gap field-wide.

## Recommended models, in build order

### 1. Baseline — One-Class SVM reject gate (build first)

- Hand-crafted features per window: per-subcarrier mean/std/skew/kurtosis,
  FFT dominant frequency + spectral entropy (gait ≈ 1–2 Hz), then PCA/LDA to
  cut dimensionality before the kernel (raw per-subcarrier windows are too
  high-dimensional for an SVM directly).
- Train OC-SVM (RBF kernel) on authorized-user windows only; anything outside
  the learned boundary → stranger.
- Optional stage 2: multiclass SVM (RBF, Platt-scaled) for per-user ID among
  knowns — only add this if per-person breakdown is ever actually needed.
- Why first: cheap, trains in seconds, tells you whether the window/feature
  config carries separable signal at all before investing in a deep model.
  zscore-per-window normalization matters more here than for a CNN, since RBF
  kernels are scale-sensitive — already covered by the current pipeline config.

### 2. Main model — CNN/TCN backbone + metric-learning head

- 1D CNN → TCN (or CNN+GRU) over raw CSI amplitude per window. TCN over LSTM
  because gait is periodic/local rather than needing long-range memory, and
  it's cheaper to train on a small dataset.
- Head: try both and compare —
  - **Prototypical network** (CAUTION-style): per-user centroid, classify +
    reject by nearest/second-nearest distance ratio.
  - **ArcFace angular-margin loss**: borrowed from face verification, same
    "authorized vs not" problem shape, tends to produce tighter/more
    separable clusters.
- Calibrate the reject threshold using only held-out known users as
  pseudo-strangers, same trick as CAUTION — no real intruder data required.

### 3. Robustness add-ons (once multi-day data exists)

- **Per-session calibration**: capture a short empty-room CSI baseline each
  day/session, normalize new windows against it before anything else.
- **Episodic cross-day training**: sample support/query from *different* days
  within each training episode — forces the embedding to generalize the way
  it would need to for a genuinely new day.
- **Domain-adversarial day-invariance** (DANN-style): small adversarial head
  predicting "which day" from the embedding; penalize the backbone for making
  that predictable.
- **Augmentation as a stand-in** until multi-day data exists: subcarrier
  dropout, amplitude jitter, time-warp, mixup across sessions.

### Alternative if per-person structure is never needed

Autoencoder / Deep SVDD trained only on authorized CSI, thresholded on
reconstruction/anomaly score. Simpler pipeline than multiclass+reject, but
usually a bit weaker than the centroid/margin approach at catching
"close-but-not-quite" strangers.

## Creative / less obvious options

Wider net beyond the safe picks above. Not all of these need to be tried —
see the "where to spend effort" note at the end of this section.

### Transformer-based

1. **Masked CSI pretraining (BERT-style, e.g. CSI-BERT2)** — self-supervised:
   mask random subcarriers/timesteps, train a transformer to reconstruct
   them, using *unlabeled* CSI (just walk-around recordings, no
   authorized/stranger labels needed). Fine-tune a small open-set head on
   top with the few labeled sessions you actually have. Best transformer
   idea for this project specifically, since labeled-data volume — not model
   capacity — is the real bottleneck.
2. **Conformer (convolution + self-attention, from speech)** — local conv
   captures footstep-impact shape, self-attention captures gait periodicity/
   cadence across the window. Gait and speech are both quasi-periodic with
   local+global structure, so this family transfers well and overfits less
   than a pure transformer on small data.
3. **Patch-based time-series transformer (PatchTST-style)** — chop the CSI
   window into patches instead of attending over every raw timestep;
   cheaper, and this family is specifically designed to hold up on small
   time-series datasets where vanilla transformers fall apart.
4. **Perceiver-style cross-attention** — handles irregular WiFi packet
   arrival times natively, so it doesn't force the fixed-window assumption
   that CNN/TCN implicitly makes.

### Other creative angles

5. **GNN over subcarriers** — model subcarriers as graph nodes with edges by
   frequency proximity/correlation, instead of assuming CNN-style local
   structure. Multipath makes subcarrier responses correlated but
   non-uniform; a GNN can represent that explicitly.
6. **Micro-Doppler radar-inspired CNN on an STFT spectrogram of CSI
   amplitude** — the RF/radar gait-identification literature is far more
   mature than WiFi CSI and solves a nearly identical problem. Treating
   CSI-amplitude-over-time as a simplified micro-Doppler spectrogram and
   borrowing radar-gait CNN architectures is a legitimate cross-field
   transfer.
7. **MAML-style daily recalibration** — instead of forcing a day-invariant
   embedding, meta-train across days-as-tasks so the model adapts in a
   handful of gradient steps to a new day using a short "walk through once"
   morning calibration. Reframes cross-day drift as few-shot adaptation
   rather than something to engineer away entirely.
8. **Sequential change-point / anomaly detection (CUSUM-style)** — reframe
   presence detection as "did the live stream just change relative to its
   own running baseline," not static per-window classification. The
   baseline updates continuously, so it tracks slow channel drift more
   gracefully than a fixed threshold.
9. **Contrastive self-supervised pretraining (SimCLR/BYOL-style)** —
   alternative/complement to #1: build positive pairs via augmentation
   (time-shift, subcarrier dropout, amplitude jitter) on unlabeled CSI,
   train an embedding invariant to those augmentations, then use a
   lightweight probe for the actual authorized/stranger decision.
10. **Gradient-boosted trees (XGBoost/LightGBM)** — a cheap sibling to the
    SVM baseline on the same hand-crafted features. Gives per-feature
    importance for free, which doubles as a diagnostic for which
    subcarriers/statistics actually carry the gait signal (ties directly
    into the calibration section below).
11. **Generative hard-negative synthesis** — train a small generator
    (VAE/GAN) on authorized-user CSI, sample near-boundary synthetic
    "almost authorized" cases, and use those to stress-test/tighten the
    open-set threshold. Real stranger data is scarce; this manufactures
    harder negatives than random noise would.
12. **Hardware-level option, not just software: add a second receiver.** A
    second cheap node turns the single link into two independent spatial
    views. The "Why Commodity WiFi Sensors Fail" paper's core finding is
    that spatial diversity — not algorithm choice — is the actual ceiling
    on commodity hardware, so this may buy more separability than any model
    change would.

**Where to spend effort:** given the real constraints (small labeled
dataset, single link, cross-day is the open problem), **#1 or #9
(self-supervised pretraining) paired with #7 or #8 (MAML recalibration or
CUSUM-style adaptive baseline)** is the most promising combo — it attacks
both actual bottlenecks (label scarcity, drift) at once, rather than just
being a fancier classifier on the same features. Treat the rest as side
experiments, not a queue to work through in order.

## Calibration & normalization — deeper options

Because the core signal is "subcarrier `k` reacts differently per person,"
calibration that operates globally (one scalar per window) risks washing out
exactly the cross-subcarrier pattern the model depends on. Options, roughly
in order of how much they respect that constraint:

1. **Per-subcarrier baseline subtraction** — capture an empty-room/no-person
   reference *per subcarrier* (not one aggregate baseline) at the start of
   each session/day, then subtract or ratio it out before windowing. This is
   the direct operationalization of the core intuition: isolate the
   person-induced perturbation on subcarrier `k` relative to that
   subcarrier's *own* static environment behavior.
2. **Per-subcarrier z-score, not a single global z-score** — normalize each
   subcarrier's amplitude independently using its own calibration-window
   mean/std. Channel frequency response isn't flat, so a global z-score over
   all subcarriers pooled together can let a strong-but-generic subcarrier
   drown out a weak-but-discriminative one.
3. **Differential/ratio CSI across time** — use frame-to-frame or
   subcarrier-to-subcarrier ratios instead of absolute per-window amplitude.
   Cancels common multiplicative gain drift (AGC changes, TX power steps)
   while preserving the relative, person-induced perturbation pattern.
4. **Phase sanitization** (only if using phase at all) — fit and remove the
   CFO/SFO-induced linear phase ramp across subcarriers before trusting any
   per-subcarrier phase feature. Raw phase on ESP32 is close to unusable
   across days without this.
5. **PCA-based common-mode removal** — compute PCA across subcarriers per
   window; the top component(s) often capture shared environmental drift
   (temperature, furniture, AGC), while the person-specific fingerprint
   lives more in the residual after removing that top component. Same idea
   as background subtraction in vision.
6. **Cross-day distribution alignment (CORAL / quantile matching)** — align
   a new day's per-subcarrier distribution to a reference day's using only
   *unlabeled* data from the new day. A lightweight substitute for full
   domain-adversarial training, and it bolts onto the SVM/GBM pipeline just
   as easily as onto a deep model.
7. **Discriminative-power-weighted calibration** — use per-subcarrier mutual
   information or the GBM feature-importance output (#10 above) to find
   which subcarriers actually carry person-specific signal, then either
   restrict features to those or calibrate them extra carefully — noisy
   calibration on a highly discriminative subcarrier costs more than on a
   low-value one.
8. **Robust per-subcarrier denoising** — median/Hampel filter along time,
   applied per subcarrier (not globally), since noise characteristics differ
   across the band — edge subcarriers are typically noisier than center
   ones.
9. **Calibration cadence** — decide between per-day (once each morning),
   per-session (each time collection starts), or continuous/adaptive
   (rolling baseline). A rolling baseline is the most robust to slow
   within-day drift and pairs naturally with the CUSUM-style adaptive
   detector (#8 in creative options); a single morning calibration is
   cheaper but goes stale by evening.

## Experiment checklist

- [ ] Run OC-SVM baseline on current 2s/0-overlap/zscore windows, held-out=room.
- [ ] Repeat OC-SVM with held-out=day once multi-day data is available.
- [ ] Train CNN/TCN + prototypical head on the same splits; compare reject
      accuracy/recall against the OC-SVM baseline.
- [ ] Swap the prototypical loss for ArcFace margin loss; compare.
- [ ] Add per-session calibration normalization; re-measure held-out=day performance.
- [ ] If the cross-day drop is still large, add domain-adversarial
      day-invariance training.
- [ ] Switch global per-window zscore to per-subcarrier zscore; check whether
      reject accuracy improves (tests the core signal intuition directly).
- [ ] Try per-subcarrier empty-room baseline subtraction at session start;
      compare against plain zscore.
- [ ] Run XGBoost/LightGBM on the same hand-crafted features as the OC-SVM
      baseline; inspect feature importance to see which subcarriers actually
      carry signal.
- [ ] Try masked-CSI or contrastive self-supervised pretraining on unlabeled
      walk-around recordings, then fine-tune a small head on the labeled
      sessions; compare against the from-scratch CNN/TCN model.
- [ ] Prototype a CUSUM-style adaptive-baseline detector as an alternative to
      a static open-set threshold; compare cross-day robustness.
