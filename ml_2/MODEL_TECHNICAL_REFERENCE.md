# ml_2 model technical reference

Every model in `ml_2`, in full technical detail: exact tensor shapes, parameter counts (computed by
actually instantiating each model, not estimated), hyperparameters, loss functions, and an ASCII
diagram of the data flow through each architecture. Read this alongside `PLAN.md` (the why) and
`README.md` (how to run things) -- this doc is the what-exactly-is-inside-the-box reference.

## 0. Shared data pipeline -- every model downstream of this

```
raw samples.npz (per session, per receiver)
  csi_flat[csi_offset[i] : csi_offset[i]+csi_len[i]]  -- ragged buffer, (imag,real) int8 pairs
        |
        v
  decode_by_bucket()  ->  amplitude = hypot(real,imag), phase = arctan2(imag,real)
        |                 bucketed by csi_len (802.11 frame type varies within a session)
        v
  dominant_bucket()   ->  the csi_len the MOST packets in this session share
        |                 confirmed empirically: channel-6 sessions are uniformly csi_len=256 bytes
        v                 = 128 raw values = LLTF(64) + HT-LTF(64) merged (lltf_en=htltf_en=
        |                 ltf_merge_en=true in every session's config, cwb=0 i.e. 20MHz HT20)
  SLICE to [:, :64]   ->  keep ONLY the legacy LLTF subcarriers (the first 64 columns) -- the
        |                 standard, present-in-every-frame-type set, per this round's explicit
        v                 instruction. Both amplitude and phase are sliced identically.
  cached per (session, receiver) as float32 .npz:  amplitude (n_packets, 64), phase (n_packets, 64),
                                                     rssi (n_packets,)
        |
        v
  build_window_index()  ->  fixed windows: window_packets=200, stride_packets=100 (50% overlap)
        |                    NOT time-normalized -- native packet rate varies ~130-600 Hz across
        |                    sessions, so window DURATION varies too (0.3-1.5s), only the packet
        |                    COUNT is fixed. Disclosed simplification, see PLAN.md.
        v
  178,040 total windows across 122 sessions, 5 dates (2026-09-15/16/17/21/22, channel 6 only --
  2026-09-09/10 excluded, confirmed different channel from raw per-packet channel_primary field)
```

**Per-(date, receiver) calibration** (`ml_2/data/calibration.py`): every window's amplitude/phase is
z-scored against that SAME date+receiver's `none` (empty-room) session statistics, per subcarrier
(64-dim mean/std vectors, never a single global scalar):

```
amp_calibrated  = (amp_raw   - baseline.amp_mean)   / (baseline.amp_std   + 1e-6)
phase_calibrated = (phase_raw - baseline.phase_mean) / (baseline.phase_std + 1e-6)
```

**Multi-receiver handling**: 2026-09-21/22 each have 3 ESP32 receivers recording in parallel
(`a4:cb:8f:d4:52:b0`, `ac:27:6e:a2:f2:78`, `ac:27:6e:a5:5b:c8`). Every model EXCEPT the explicit
`ReceiverFusionTransformer` treats each receiver's stream as its own fully independent training row
(not fused, not collapsed to one "representative" receiver) -- this is why those 2 dates contribute
roughly 3x the rows of the single-receiver dates. Leakage-safe splits group by `session_dir` only, so
a session's 3 receivers always move together across train/test (never split across the boundary).

**Splits used everywhere** (`ml_2/data/splits.py`):
- `leave_one_unauthorized_person_out`: holds out ONE real stranger identity entirely (9 people:
  abdul/divya/harshitha/kishore/manas/promoda/siva/sumanth/vishwa) + a fixed 25% slice of authorized
  and `none` sessions -- the genuine open-set check.
- `leave_one_day_out`: holds out one whole date, trains on the other 4 -- the genuine cross-day check.
- `session_disjoint_kfold` (5-fold): closed-set sanity number, reported for context only.

**Evaluation: window-level AND session-level, always both** (`ml_2/data/metrics.py::session_level_metrics`).
A pooled/window-level AUROC is measured over windows that overlap 50% WITHIN a session -- highly
correlated, near-duplicate samples, not independent draws. A fold's window-level number can look
strong (or weak) purely because a handful of sessions happen to contribute thousands of windows
each, not because the model actually separates the classes on genuinely different sessions. Every
model's evaluation therefore ALSO collapses to one decision per session (mean probability + majority
vote across that session's windows) and recomputes accuracy/AUROC at that level -- printed as
`SESSION-LEVEL` alongside the pooled `WINDOW` number, and logged as separate `session_accuracy`/
`session_auroc` columns. **The session-level number is the one to trust; the window-level number is
reported for transparency, not as the headline metric.** A worked synthetic check confirmed the two
can diverge substantially in either direction depending on how errors are distributed across sessions
of different sizes -- this is exactly the failure mode being guarded against, not a hypothetical.

**Handcrafted feature vector** (classical models -- SVM/GMM, GBM, CUSUM, VAE): per-subcarrier
mean/std/skew/kurtosis, for BOTH amplitude and phase, plus RSSI mean/std:

```
64 subcarriers x 4 stats (mean,std,skew,kurt) x 2 channels (amp,phase)  =  512
+ rssi_mean, rssi_std                                                   =  2
                                                                    TOTAL = 514-dim feature vector
```

---

## 1. Classical models (`models/svm_gait.py`, `gbm_baseline.py`) -- operate on the 514-dim feature vector

### 1a. OneClassGaitSVM (`oc_svm`)

```
514-dim feature vector -> StandardScaler -> OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
                                                    |
                                             decision_function(x)  (signed distance to the
                                             learned boundary; +ve = inlier/authorized-like)
```
- **Kernel: exactly ONE kernel, RBF (Gaussian)** -- `K(x, x') = exp(-gamma * ||x - x'||^2)`. No
  polynomial/linear/sigmoid alternative is tried; RBF was picked because the feature space (raw
  per-subcarrier moments) has no reason to be linearly separable, and CAUTION/Wii-style CSI
  authentication work uses RBF/Gaussian kernels for the same reason.
- **Regularization / capacity control (2 knobs, both scikit-learn defaults except `nu`):**
  - `nu=0.1` -- the ν-SVM regularization parameter. Simultaneously an upper bound on the fraction
    of TRAINING points allowed to be margin errors/outliers and a lower bound on the fraction that
    become support vectors. `nu=0.1` means the boundary is fit expecting ~10% of the authorized
    training windows to sit on the wrong side -- this is the single biggest lever if this model's
    reject boundary is too tight or too loose (see the training log: session-level AUROC ~0.35,
    worse than chance, is the first thing worth revisiting by sweeping `nu`).
  - `gamma='scale'` -- RBF kernel width, set automatically to `1 / (n_features * X.var())` from the
    training data rather than a fixed constant. Controls how far a single training point's
    influence reaches; not swept here.
  - `StandardScaler` (zero-mean, unit-variance per feature) is itself a de facto regularizer for
    an RBF kernel -- without it, features with larger raw scale (e.g. `amp_mean` vs `phase_kurt`)
    would dominate the distance computation `gamma * ||x - x'||^2` regardless of `gamma`.
- **Trained on**: authorized-only rows (label=="authorized"), subsampled to <=6000 rows/fold
  (RBF-kernel SVMs are ~O(n^2)-O(n^3); this dataset's folds have 100k+ rows, which would take
  hours-to-never without subsampling). See `MAX_SVM_TRAIN_SAMPLES` in `train_svm_gait.py`.
- **No labels needed for "unauthorized"** -- this is the whole point of a one-class model.
- **Parameters**: no learned weights in the traditional sense -- the model IS its support vectors
  (up to 6000 of the 514-dim training points) plus their dual coefficients. Not a fixed parameter
  count like a neural net.

### 1b. BinaryGaitSVM (`binary_svc`)

```
514-dim feature vector -> StandardScaler -> SVC(kernel='rbf', C=1.0, gamma='scale',
                                                  class_weight='balanced', probability=True)
                                                    |
                                             predict_proba(x)[:, class=1]
```
- **Kernel: ONE kernel, RBF** -- same `K(x,x') = exp(-gamma*||x-x'||^2)` as 1a, same reasoning.
- **Regularization / capacity control (3 knobs):**
  - `C=1.0` -- the standard SVM regularization constant (inverse strength): the soft-margin
    objective is `1/2 ||w||^2 + C * sum(hinge_loss)`. `C=1.0` is scikit-learn's default, i.e. a
    balanced trade-off, not tuned for this data -- a smaller `C` would widen the margin/underfit
    more, a larger `C` would fit training points more tightly (more support vectors, more risk of
    overfitting the 6000-row subsample).
  - `gamma='scale'` -- same auto-computed RBF width as 1a.
  - `class_weight='balanced'` -- reweights the hinge loss by `n_samples / (n_classes * class_count)`,
    i.e. the minority class (authorized windows are outnumbered by unauthorized+none pooled
    together) gets a proportionally larger penalty for misclassification. This is a rebalancing
    mechanism, not a smoothness regularizer, but it directly changes what the margin optimizes for.
  - `StandardScaler`, same role as 1a.
- **Trained on**: ALL rows (authorized=1 vs unauthorized-or-none=0), subsampled to <=6000/fold.
- Ordinary closed-set classifier -- included as the direct SVM-vs-RandomForest-style baseline
  comparison, NOT an open-set model (structurally cannot say "unknown").
- `probability=True` internally runs a 5-fold Platt-scaling calibration on top of the base fit --
  this is why SVC folds take noticeably longer than the OC-SVM's.

### 1c. GaitGmmSvmTwoStage (`gmm_svm_wii`) -- Wii (Sensors 2017) replica

```
Stage 1 (reject gate):
  authorized-only features -> StandardScaler -> GaussianMixture(n_components=3)
                                                        |
                                                 score_samples(x)  (log-likelihood under the
                                                 fitted mixture -- the genuine/authorized score)
                                                        |
                                        threshold = 5th percentile of TRAINING log-likelihoods
                                        (so ~5% of genuine authorized windows are themselves
                                         borderline-rejected -- calibrated on knowns only)

Stage 2 (identity, only reached if stage 1 accepts):
  same authorized-only features -> SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
                                                        |
                                                 predict(x) -> 'anjali' or 'barath'
```
- **Stage 1 has NO kernel at all** -- `GaussianMixture` is a density-estimation model (a weighted
  sum of `n_components=3` full-covariance Gaussians fit via EM), not a kernel method. Regularization
  here is `n_components=3` (capacity: 3 Gaussians, not 1 or 10 -- scikit-learn's default
  `covariance_type='full'` means each component gets its own full 514x514 covariance matrix, the
  least-regularized covariance option scikit-learn offers vs `'diag'`/`'tied'`/`'spherical'`) plus
  the `gmm_reject_percentile=5.0` threshold placement (analogous role to `nu` in 1a: sets what
  fraction of genuine training data is treated as the acceptable false-reject rate).
- **Stage 2 kernel: ONE kernel, RBF**, same `C=1.0`/`gamma='scale'` as 1b (no `class_weight` here --
  stage 2 only ever sees authorized-only rows split by identity, which per this dataset is close to
  balanced between anjali/barath already).
- Both stages trained on the SAME subsampled (<=6000) authorized-only rows.
- Stage 2 is skipped (falls back to stage-1-only) if a fold's training data has <2 identities
  present (can happen on some leave-one-day-out folds).

### 1d. GaitGBM (XGBoost)

```
514-dim feature vector -> XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                          device='cuda' if available else 'cpu')
                                                    |
                                             predict_proba(x)[:, class=1]
```
- **No kernel** -- gradient-boosted decision trees, not a kernel method.
- **Regularization / capacity control (explicit + implicit defaults):**
  - `max_depth=4` -- explicit capacity cap per tree (each tree can branch at most 4 levels deep).
  - `learning_rate=0.1` (a.k.a. `eta`) -- shrinks each tree's contribution to the ensemble,
    the standard boosting regularizer: smaller values need more trees but generalize better.
  - `n_estimators=200` -- ensemble size; combined with `learning_rate=0.1` this is the
    bias/variance trade-off knob actually being controlled here.
  - **Not explicitly set, so XGBoost's internal defaults apply** (the sklearn wrapper reports
    these as `None`, which means "let the C++ booster use its own default", not "off"):
    `reg_lambda=1` (L2 penalty on leaf weights -- ridge-style shrinkage, ON by default),
    `reg_alpha=0` (L1 penalty -- off), `gamma=0` (no minimum loss reduction required to split, i.e.
    no extra pruning beyond `max_depth`), `subsample=1.0` and `colsample_bytree=1.0` (no row/column
    subsampling -- every tree sees the full feature set and, unlike the SVMs, the FULL fold's rows).
- **Trained on**: the FULL fold (no subsampling -- gradient-boosted trees scale near-linearly in
  n_samples, unlike kernel SVMs).
- 200 trees x depth-4 -> up to 200 x 2^4 = 3200 leaf nodes total (typically far fewer in practice,
  trees rarely reach max depth on every path).
- `feature_importances_` is read out after training as a diagnostic (which of the 514 features
  actually carry signal) -- see results log for the actual ranking.

---

## 2. CUSUM adaptive detector (`models/cusum_detector.py`) -- sequential, not i.i.d.

```
Input per window: 64-dim vector (amp_mean per subcarrier ONLY -- first 64 columns of the 514-dim
                   feature vector; std/skew/kurt and phase are not used by this model)

State carried across windows WITHIN one session, in time order:
  running_mean (64-dim, EMA-updated)         <- initialized from authorized-only training data
  g_pos (scalar, the CUSUM cumulative statistic)

Per window x:
  z = (x - running_mean) / sqrt(baseline_var)               # per-subcarrier z-score
  deviation = mean(z^2)                                      # scalar: how far from baseline
  g_pos = max(0, g_pos + deviation - k)          k=0.5 (slack)
  score = -g_pos                                 (higher = more baseline-consistent = more genuine)
  running_mean = (1-alpha)*running_mean + alpha*x            alpha=0.02
  if g_pos > h:  g_pos = 0                        h=8.0 (reset after a flagged change point)
```
- **No neural network, no fixed parameter count** -- 3 scalar hyperparameters (`alpha`, `k`, `h`)
  plus the 64-dim running mean/var state.
- **Known issue** (see training log): the adaptive baseline drifts toward whatever it's CURRENTLY
  seeing, so on a long single-label session it can "normalize" to the wrong class partway through --
  measured AUROC (0.16-0.25) is well below chance, consistent with a systematic direction/drift
  problem rather than random noise. Flagged as an open issue, not silently hidden.

---

## 3. Embedding backbone (`models/backbone.py`) -- shared by prototypical / ArcFace / contrastive

### `BranchEncoder` (one instance per channel: amplitude, phase)

```
input (B, 200, 64)
      |
      v
Linear(64 -> 32)                          input_proj
      |
      v
+ SinusoidalPositionalEncoding(32)        added elementwise, not learned
      |
      v
TransformerEncoderLayer(                   num_layers=1
   d_model=32, nhead=4,                    (RESEARCH_NOTES-style finding: shallow/narrow beats
   dim_feedforward=64,                      deep on this dataset size)
   dropout=0.2, batch_first=True,
   norm_first=False )                      post-LN (norm AFTER the residual, original-paper style)
      |
      v
output (B, 200, 32)   -- one 32-dim token per timestep
```
Inside one `TransformerEncoderLayer` (post-LN): `x = LayerNorm(x + Dropout(SelfAttn(x)))`, then
`x = LayerNorm(x + Dropout(FFN(x)))`, where `SelfAttn` is 4-head scaled dot-product attention
(each head: 32/4=8-dim) and `FFN` is `Linear(32,64) -> ReLU -> Dropout -> Linear(64,32)`.

**Regularization used by EVERY neural model in this document (backbone-based or not), stated once
here rather than repeated per section:**
- **Dropout=0.2 is the ONLY regularization mechanism** -- applied inside every
  `TransformerEncoderLayer`/`CrossAttentionBlock` (this section, and transformers.py/
  receiver_fusion.py which reuse it), and independently set to the same `0.2` in `SubcarrierGNN`
  and `RadarInspiredCNN`'s own classifier heads (models 7 and 8 below).
- **No weight decay anywhere.** Every `torch.optim.Adam(...)` and `torch.optim.SGD(...)` call in
  this codebase (`gpu_utils.py`, `train_prototypical.py`, `train_arcface.py`,
  `train_contrastive.py`, `train_receiver_fusion_gpu.py`, `maml_wrapper.py`,
  `generative_hard_negative.py`) omits `weight_decay`, so it defaults to `0` -- i.e. there is no
  L2 penalty on the WEIGHTS of any neural model here, unlike the SVMs above (whose `C`/`nu` ARE a
  form of weight-norm regularization) or GBM (whose `reg_lambda=1` is on by default). If any
  neural model here overfits, dropout and early stopping (fixed epoch counts, not validation-based)
  are the only levers currently in place -- adding `weight_decay` to the optimizer is the natural
  next lever, not yet tried.
- **Adam betas/eps are scikit-learn -- no, PyTorch -- defaults** (`betas=(0.9, 0.999)`, `eps=1e-8`)
  everywhere Adam is used; only `lr` is ever set explicitly (`1e-3` for backbone/head training,
  `1e-2` for the SGD-based MAML inner-loop steps, see model 11).

### `DualBranchEmbedding` (the full backbone)

```
amplitude (B,200,64) --> BranchEncoder --> (B,200,32) --> mean over time --> (B,32)  \
                                                                                       concat -> (B,64)
phase     (B,200,64) --> BranchEncoder --> (B,200,32) --> mean over time --> (B,32)  /
                                                                                       |
                                                                                       v
                                                              Linear(64->32) -> ReLU -> Dropout
                                                                                       |
                                                                                       v
                                                                            Linear(32->32)
                                                                                       |
                                                                                       v
                                                                        L2-normalize  (B,32)
                                                                     <- final embedding, ||e||=1
```
**Total parameters: 24,384** (two independent BranchEncoders, NOT weight-shared -- amplitude and
phase get their own weights, unlike the receiver-fusion model below which DOES share weights).

---

## 4. Prototypical-network open-set head (`models/prototypical.py`)

Uses the `DualBranchEmbedding` backbone above (24,384 params) -- no extra learned parameters, the
"head" is pure geometry (centroids + distances), not a weight matrix.

```
Training (per fold):
  authorized-only rows -> split 80/20 into "centroid" / "calibration" subsets (seeded)
  for each epoch (x8), for each batch of 64 from the centroid subset:
        if <2 classes present in this batch: skip
        embed all 64 windows -> (64, 32)
        episodic_split(labels) -> support_idx, query_idx   (~half/half per class in the batch)
        centroids = mean of SUPPORT embeddings per class    (anjali centroid, barath centroid) (2,32)
        loss = CrossEntropy( -||query_emb - centroid||^2 ,  query_label )     <- eq. 2-3, CAUTION
        Adam(lr=1e-3), backprop through backbone + the centroid computation itself

Calibration (after training, still no stranger data):
  embed the held-out 20% "calibration" subset (still authorized-only)
  distance_ratio = d(nearest centroid) / d(second-nearest centroid)     <- CAUTION eq. 4
  threshold = 95th percentile of these genuine ratios  (accepts ~95% of held-out authorized windows)

Inference on a real test window:
  embed -> nearest_centroid, ratio = d1/d2
  genuine_score = -ratio        (lower ratio = more confidently ONE identity = more genuine)
  accept if ratio <= threshold
```
- **Classes used for centroids**: `anjali`, `barath` only -- `non_auth` windows never participate
  in training or centroid computation (CAUTION's "no intruder data needed" property).
- **Batch size**: 64. **Epochs**: 8 (CLI default). **Optimizer**: Adam, lr=1e-3.
- **Measured result**: open-set AUROC 0.640, but cross-day AUROC 0.497 (chance) -- the embedding
  does not currently generalize across days.

---

## 5. ArcFace open-set head (`models/arcface.py`)

Same `DualBranchEmbedding` backbone (24,384 params) + a small `ArcFaceHead`:

```
ArcFaceHead(embed_dim=32, n_classes=2, scale=30.0, margin=0.3):
    weight: (2, 32) learned matrix, one row per identity           <- 64 parameters total

Training:
  cosine = normalize(embedding) @ normalize(weight).T          (B, 2) cosine similarities
  theta  = arccos(cosine)
  target_logit = cos(theta + margin)          margin=0.3 radians, ONLY for the true class's column
  logits = cosine*(1-one_hot) + target_logit*one_hot,  then * scale (30.0)
  loss = CrossEntropy(logits, label)

Inference (open-set score):
  genuine_score = max(cosine similarity to any class's weight row)     <- margin NOT applied at eval
```
- **Total new parameters vs. prototypical**: only 64 (the 2x32 weight matrix) -- ArcFace and
  prototypical are otherwise parameter-identical, isolating the loss-function choice as the only
  difference between the two experiments.
- Same training config as prototypical: batch 64, epochs 8, Adam lr=1e-3.
- **Measured result**: open-set AUROC 0.494, cross-day AUROC 0.499 -- both at chance. Underperforms
  prototypical on this dataset/config as run.

---

## 6. Transformer classifier zoo (`models/transformers.py`)

All three share the `BranchEncoder` building block (32-dim, 4-head, 1-layer) from section 3, but
classify directly (softmax) instead of building an embedding space.

### 6a. WhoFiTransformer -- **amplitude only**

```
amplitude (B,200,64) -> BranchEncoder -> (B,200,32) -> mean over time -> L2-normalize -> (B,32)
                                                                                |
                                                                                v
                                                                    Linear(32 -> n_classes=2)
```
**10,690 parameters.** Phase is passed into `forward()` but immediately discarded (`del phase`) --
this is the "cheap floor" model, deliberately minimal.

### 6b. DualBranchTransformer -- amplitude AND phase, late fusion

```
amplitude -> BranchEncoder_amp -> mean -> (B,32)  \
                                                     concat -> (B,64) -> Linear(64,32)+ReLU+Dropout
phase     -> BranchEncoder_pha -> mean -> (B,32)  /                            |
                                                                                v
                                                                    Linear(32 -> n_classes)
```
**23,394 parameters** (roughly 2x WhoFi, since it has two independent branches instead of one).

### 6c. CrossAttentionTransformer -- amplitude and phase bidirectionally cross-attend

```
amp_tokens   = BranchEncoder_amp(amplitude)              (B,200,32)
phase_tokens = BranchEncoder_pha(phase)                  (B,200,32)
                    |                          |
                    v                          v
      CrossAttentionBlock(query=amp, key/value=phase) -> amp_cross   (B,200,32)
      CrossAttentionBlock(query=phase, key/value=amp) -> phase_cross (B,200,32)
                    |                          |
              mean over time              mean over time
                    |                          |
                    +----------- concat -------+  -> (B,64) -> Linear(64,32)+ReLU+Dropout -> Linear(32,n_classes)
```
Each `CrossAttentionBlock` is a full transformer sub-block, not just an attention op:
`x = LayerNorm(x + Dropout(CrossAttn(query=x, key=context, value=context)))`, then
`x = LayerNorm(x + Dropout(FFN(x)))`.
**40,482 parameters** -- the largest of the three (two self-attention branches + two full
cross-attention blocks, each with their own FFN).

**Training for all three** (`training/train_transformer_gpu.py`): task = `auth_vs_nonauth` (binary),
CrossEntropyLoss, Adam lr=1e-3, batch 64, epochs=6 (CLI default), seed=0. Evaluated under
leave-one-unauthorized-person-out and leave-one-day-out, same as every other model.
**Measured result so far**: WhoFi's first open-set fold hit AUROC 0.975 -- currently the best single
result in the entire sweep.

---

## 7. Subcarrier GNN (`models/gnn_subcarrier.py`) -- amplitude AND phase

```
Node features per subcarrier (64 nodes total):
    [amp_mean_over_time, amp_std_over_time, phase_mean_over_time, phase_std_over_time]   (4-dim/node)

Fixed adjacency (NOT learned): each subcarrier connected to its +/-4 nearest neighbors in frequency,
row-normalized. Average degree: 8.7 neighbors/node (64x64 adjacency matrix, banded).

(B, 64 nodes, 4 features)
        |
        v
GCNLayer: h = ReLU( Linear(4->32)( adjacency @ node_features ) )       <- Kipf-style GCN propagation
        |
        v  Dropout(0.2)
        v
GCNLayer: h = ReLU( Linear(32->32)( adjacency @ h ) )
        |
        v
mean over the 64 nodes -> (B, 32)
        |
        v
Linear(32 -> n_classes)
```
**1,282 parameters** -- by far the smallest model in the whole sweep (the adjacency matrix is a
fixed buffer, not learned weights; only the two `Linear` layers inside the GCN layers, plus the
final classifier, have gradients).

---

## 8. Radar-inspired CNN (`models/radar_cnn.py`) -- amplitude only

```
amplitude (B, 200, 64)
        |
        v
mean over subcarriers -> (B, 200)      <- collapses the 64-subcarrier axis; the micro-Doppler proxy
        |                                  signal is the AMPLITUDE MODULATION over time, not
        v                                  per-subcarrier structure (deliberately different
torch.stft(n_fft=32, hop_length=4,          inductive bias from every other model here)
           window=hann(32), center=True)
        |
        v
complex spectrogram (B, 17, 51)         <- 17 = n_fft//2+1 frequency bins, 51 = time frames
        |
        v
log(1 + |spectrogram|) -> unsqueeze channel dim -> (B, 1, 17, 51)
        |
        v
Conv2d(1->16, k=3, pad=1) -> ReLU -> MaxPool2d(2)          -> (B, 16, 8, 25)
        |
        v
Conv2d(16->32, k=3, pad=1) -> ReLU -> AdaptiveAvgPool2d((4,4))  -> (B, 32, 4, 4)
        |
        v
flatten -> (B, 512) -> Dropout(0.2) -> Linear(512 -> n_classes)
```
**5,858 parameters.** The only model in the sweep that takes a frequency-domain-of-time transform
(as opposed to a frequency-domain-of-subcarrier one, which is what "subcarrier" already is) --
borrowed from micro-Doppler radar gait-ID literature, where the ~1-2 Hz walking-cadence signature
shows up directly as spectrogram energy.

---

## 9. 3-receiver fusion (`models/receiver_fusion.py`) -- amplitude only, scoped to 2026-09-21/22

```
receiver 1 amplitude (B,200,64) -\
receiver 2 amplitude (B,200,64) --> SAME shared BranchEncoder (weight-shared, not 3 separate copies)
receiver 3 amplitude (B,200,64) -/          |
                                             v
                                  each -> mean over time -> L2-normalize -> (B,32) per receiver
                                             |
                                  concat all 3 -> (B, 96)
                                             |
                                             v
                              Linear(96->32) -> ReLU -> Dropout -> Linear(32 -> n_classes)
```
**13,794 parameters** -- notably FEWER than DualBranchTransformer (23,394) despite fusing 3 streams
instead of 2, because the branch encoder is weight-SHARED across all 3 receivers (appropriate given
only 2 days' worth of 3-receiver data -- far less than the 5-day pool used everywhere else).
Windows from the 3 receivers are aligned by POSITION within each session (the Nth window from each),
not exact timestamp -- the 3 receivers have independent, unsynchronized onboard clocks.

---

## 10. Contrastive self-supervised pretraining (`models/contrastive_pretrain.py`)

Uses the same `DualBranchEmbedding` backbone (24,384 params). Two-stage:

**Stage 1 -- pretrain on ALL 178,040 windows, labels ignored entirely:**
```
for each window x in a batch of 64:
    view1 = augment(x):  subcarrier dropout (p=0.15, zeroes random subcarrier columns)
                        + amplitude jitter (gaussian noise, std=0.05)
                        + random time-shift (+/-10 packets, torch.roll)
    view2 = augment(x)   <- independently re-sampled augmentation of the SAME window
    z1 = backbone(view1),  z2 = backbone(view2)      both (B, 32), L2-normalized
    NT-Xent loss (temperature=0.2):
        sim = normalize(concat(z1,z2)) @ normalize(concat(z1,z2)).T / 0.2     (2B, 2B)
        each view's positive = its OWN other view; every other row = a negative
        loss = CrossEntropy(sim_with_diagonal_masked, positive_indices)
10 epochs (CLI default), Adam lr=1e-3, over the FULL unlabeled pool.
```

**Stage 2 -- per fold, backbone FROZEN, only a linear probe trains on that fold's labeled data:**
```
LinearProbe: Linear(32 -> n_classes)          <- 66 parameters for n_classes=2 (32*2 + 2 bias)
embed the fold's train windows with the frozen backbone -> (N, 32)
train probe: full-batch (all training embeddings at once, no mini-batches -- cheap since it's
             just a 32->2 linear layer), Adam lr=1e-2, 10 epochs (CLI default)
```
- The point: attack label scarcity by learning representations from cheap UNLABELED data first, so
  the actual supervised step only has to fit 66 parameters per fold, not 24,384.

---

## 11. Reptile daily-recalibration (`models/maml_wrapper.py`)

Base model: `WhoFiTransformer` (10,690 params, section 6a) -- NOT the embedding backbone.

```
META-TRAINING (once per held-out-day fold, using the OTHER 4 days as day-tasks):
  meta_weights = WhoFiTransformer() initial random weights
  for meta_epoch in range(10):                              <- meta_epochs (CLI default)
    for each of the 4 non-held-out days:
        fast_model = deepcopy(meta_weights)
        for inner_step in range(5):                          <- inner_steps
            batch = next(that day's DataLoader)
            loss = CrossEntropy(fast_model(amp,phase), label)
            SGD(fast_model.parameters(), lr=1e-2).step()      <- inner_lr, plain SGD not Adam
        meta_weights += 0.5 * (fast_model.weights - meta_weights)     <- meta_lr=0.5, Reptile update
                                                                          (Nichol et al. 2018)

ADAPT-TO-DAY (deployment-time simulation, on the HELD-OUT day):
  calib_slice = random 20% of the held-out day's windows        <- CALIB_FRACTION
  adapted_model = deepcopy(meta_weights)
  for inner_step in range(10):                                   <- inner_steps (adapt call)
      batch = next(calib_slice's DataLoader)
      SGD(adapted_model.parameters(), lr=1e-2).step()
  evaluate adapted_model on the REMAINING 80% of the held-out day
```
- **First-order approximation of MAML** (no second-derivative through the inner-loop steps) --
  deliberately chosen for cost; the core idea (meta-train a starting point that adapts fast) is the
  same as full MAML.
- **This is the one model in the sweep that gets ANY same-day data before being tested** -- a 20%
  calibration slice. Reported under its own `cross_day_loo_with_calib` split type, never blended
  with the zero-peek `cross_day_loo` numbers from every other model.

---

## 12. Generative hard-negative VAE (`models/generative_hard_negative.py`)

```
FeatureVAE(input_dim=514, latent_dim=16, hidden_dim=64):

  encoder:  Linear(514 -> 64) -> ReLU
  mu:       Linear(64 -> 16)
  logvar:   Linear(64 -> 16)
  decoder:  Linear(16 -> 64) -> ReLU -> Linear(64 -> 514)

  TOTAL: 69,538 parameters (the largest single model, dominated by the 514<->64 input/output layers)

forward(x):
  h = encoder(x)
  mu, logvar = mu_head(h), logvar_head(h)
  z = mu + exp(0.5*logvar) * randn_like(mu)          <- reparameterization trick
  x_reconstructed = decoder(z)
  loss = MSE(x_reconstructed, x) + 0.01 * KL(N(mu,var) || N(0,I))

Trained on: authorized-only features (FULL set, not subsampled -- a small MLP full-batch forward
            pass is cheap even at 40k+ rows, unlike the SVM kernel computations).
100 epochs (module default), Adam lr=1e-3, full-batch (no mini-batching).

Hard-negative sampling:
  z ~ N(0, I) * push_factor(1.8)         <- scale the latent sample's norm OUTWARD by 1.8x before
  x_hard_negative = decoder(z)               decoding, so it lands past the typical authorized
                                              latent radius ("almost authorized but not quite")
```
- **Regularization: the `0.01` KL weight IS the regularizer** here, not an afterthought -- it's
  what keeps the latent space close to `N(0,I)` instead of collapsing to a handful of memorized
  points, which matters twice over: (a) it's what makes `z ~ N(0,I)` a meaningful sampling
  distribution for hard-negative generation at all, and (b) it's the only thing preventing the
  autoencoder from just memorizing the (relatively small) authorized-only training set. No dropout,
  no weight decay in this model -- the KL term is the whole regularization story. `push_factor=1.8`
  is a separate, non-regularization knob (how far outside the learned manifold to sample).
- **The comparison SVC** (real-authorized vs synthetic-hard-negative) reuses the exact same kernel
  and regularization as model 1b: `SVC(kernel='rbf', class_weight='balanced', probability=True)`
  with scikit-learn's default `C=1.0`, `gamma='scale'` (not explicitly overridden in this script).
  Both this SVC and the baseline `OneClassGaitSVM` it's compared against are fit on the SAME
  `MAX_SVM_TRAIN_SAMPLES=6000`-subsampled authorized rows as models 1a-1c, for the same libsvm
  scaling reason.
Used two ways in `training/train_generative_hard_negative.py`: (1) diagnostic -- what fraction of
synthetic hard negatives does a plain OC-SVM accept; (2) an SVC trained on real-authorized(+1) vs
synthetic-hard-negative(-1), compared against the plain OC-SVM on the SAME real held-out strangers.

---

## 13. Training infrastructure (added after the first pass -- not a model, but changes how every
model above is actually executed)

**Multi-GPU fold parallelism** (`training/gpu_utils.py::run_folds_parallel`, `FOLD_DEVICES`). Every
model whose `run_fold`/per-fold training function creates a FRESH model from scratch (models 4, 5,
6, 7, 8, 9, 11 -- prototypical, ArcFace, the transformer zoo, GNN, radar-CNN, receiver fusion,
MAML) now runs its independent folds concurrently, one per physical GPU, via a `ThreadPoolExecutor`
sized to `torch.cuda.device_count()` (2 on this box). Threads, not processes: PyTorch releases the
GIL during the actual CUDA kernel launches, so 2 threads pinned to `cuda:0`/`cuda:1` genuinely run
in parallel without multiprocessing's CUDA-context/pickling overhead. Confirmed empirically (not
just asserted): a 4-item dummy-GPU-workload test showed items round-robin across both devices with
near-linear speedup, and a real WhoFi fold run showed both GPUs at ~20-22% utilization
*simultaneously* (previously 0% on the idle one).

**Deliberately NOT parallelized: model 10 (contrastive pretrain+probe).** Its per-fold step reuses
ONE shared, already-pretrained, frozen backbone across every fold. `model.to(device)` mutates the
model's parameters in place -- if two folds on different devices called it concurrently on the same
shared object, one fold's `.to('cuda:1')` would yank the tensors out from under the other fold
mid-forward-pass on `'cuda:0'`. Fixing this properly would mean deep-copying the frozen backbone
once per device; skipped for now as not worth the risk for one stage. Model 10 still benefits from
the DataLoader fixes below, just not from 2-GPU fold concurrency.

**CPU headroom, not full-core claim.** `WORKERS_PER_FOLD = max(1, min(4, (os.cpu_count() - 4) //
len(FOLD_DEVICES)))` -- reserves 4 cores for normal machine use, then splits what's left evenly
across however many folds are running concurrently, capped at 4 DataLoader workers per fold either
way (more workers than that just adds process overhead for windows this small).

**A real Python-version bug this surfaced and fixed**: Python 3.14 changed the default
`multiprocessing` start method on Linux from `fork` to `forkserver`, which requires DataLoader
worker arguments to be picklable. The per-(date,receiver) calibration closures used throughout
`ml_2/data/calibration.py` aren't cleanly picklable under `forkserver`, so `num_workers>0` failed
with a pickle error the FIRST time it was actually exercised (caught by an integration test before
being trusted in the long unattended run, not discovered after the fact). Fixed by passing
`multiprocessing_context="fork"` explicitly to every `DataLoader(...)` call in the codebase (14
call sites), which restores the old fork-without-pickling behavior for just these workers.

---

## Parameter count summary (computed, not estimated)

| Model | Params | Amp/Phase | Notes |
|---|---|---|---|
| SubcarrierGNN | 1,282 | both | smallest net; adjacency is fixed, not learned |
| ArcFaceHead (on top of backbone) | 64 | -- | isolates loss-function choice vs. prototypical |
| WhoFiTransformer | 10,690 | amp only | the "cheap floor" |
| ReceiverFusionTransformer (3 recv) | 13,794 | amp only | weight-SHARED branch across receivers |
| RadarInspiredCNN | 5,858 | amp only | only model with a time-domain STFT |
| DualBranchTransformer | 23,394 | both | ~2x WhoFi (two independent branches) |
| DualBranchEmbedding (backbone) | 24,384 | both | used by prototypical/ArcFace/contrastive |
| CrossAttentionTransformer | 40,482 | both | largest classifier (2 self-attn + 2 cross-attn blocks) |
| FeatureVAE | 69,538 | both (via features) | largest overall; dominated by 514-dim in/out layers |

## Window/data-volume quick reference

| Quantity | Value |
|---|---|
| Window size | 200 packets (duration varies with native packet rate, ~0.3-1.5s) |
| Stride | 100 packets (50% overlap) |
| Subcarriers kept | 64 (legacy/LLTF only, sliced from a 128-wide merged buffer) |
| Channels | amplitude + phase (model-dependent which/both are used, see above) |
| Total windows | 178,040 |
| Sessions | 122 (across 5 dates, channel 6 only) |
| Feature vector (classical models) | 514-dim (64 subcarriers x 4 stats x 2 channels + 2 rssi) |
| SVM/GMM per-fold training cap | 6,000 rows (random subsample; folds otherwise have 100k+ rows) |
