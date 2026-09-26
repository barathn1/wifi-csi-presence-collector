# Home Model & Signature+EVM Architecture Notes

Covers two separate model families currently on this branch:

1. **Home model (v1 / v2)** — the deployable RandomForest identity classifier used by
   `ml/inference/score_live_session.py` (`HOME_MODEL_V2_TESTING_RUNBOOK.md`). **This is "the
   home model."**
2. **Signature + EVM** — a separate, experimental open-set (accept/reject) WhoFi-style
   embedding model in `ml/models/signature_evm.py`, trained by
   `ml/training/train_signature_evm_day3day4.py`. Not called "home model" anywhere in the repo,
   not wired into `score_live_session.py` — included here because it's the one place EVM is used.

They are unrelated pipelines trained on different feature representations — the home model
never touches a neural network, and the signature model never touches a RandomForest. Each
section below states **what** was done, **how** it works mathematically, and **why** that
choice was made over the alternatives, and tags every reusable technique with its published/
common name so it's traceable to the wider literature, not just this codebase.

---

## 1. Home Model (RandomForest, clip-recipe features)

Source: `ml/training/train_home_model.py`, feature/clip logic in
`ml/training/run_clip_recipe_identity.py`. Two-class closed-set identification: **anjali vs
barath** only — it always outputs one of those two names, never "unknown."

### 1.1 Preprocessing pipeline (per session → per clip → feature vector)

| Step | What | How (math) | Why |
|---|---|---|---|
| Amplitude | Convert raw I/Q to a real-valued signal | `amplitude = sqrt(real² + imag²)` per subcarrier, per packet — this is just the modulus of the complex CSI value | CSI is inherently complex (amplitude + phase); amplitude alone is the conventional, more robust channel for occupancy/identity work since phase is highly sensitive to hardware sampling-offset noise on cheap radios like the ESP32 |
| Packet cleaning | Drop bad packets | Remove `first_word_invalid` packets and duplicate-timestamp packets (`clean_packet_mask`, `drop_duplicate_timestamps`) | Corrupted/duplicated packets would otherwise silently corrupt the per-clip statistics below |
| Dominant CSI length | Keep one consistent packet shape per session | Bucket packets by `csi_len`; keep only the most-frequent (mode) length | A session occasionally contains a few malformed frames with a different length; mixing lengths would break the fixed-width feature vector |
| Clip windowing | Turn a variable-length session into fixed-size analysis units | **3.0 s clips, 1.0 s stride** (67% overlap) over the cleaned timeline | A short enough window to react quickly in a live test, but long enough to gather 100+ packets per clip for stable moment estimates; overlapping (stride < length) trades redundant compute for smoother, more numerous training samples |
| Coverage gate | Reject noisy/underfilled clips | Drop a clip if it has `< 0.5 × expected_rate_hz × 3.0 s` packets | If Wi-Fi congestion or a channel hop starves a clip of packets, its moments would be unstable; 50% is a coverage floor, not a hard theoretical constant |
| Per-clip reduction | Compress a variable-length clip into a fixed-length vector | See moment formulas below, computed **per subcarrier**, over all packets in the clip | A clip has a variable packet count (packet rate isn't perfectly constant), but a classifier needs a fixed-length input — reducing to summary statistics avoids needing to resample every clip to one common sequence length |

**Moment formulas** (`ml/data_pipeline/features.py::_moments`, computed independently per
subcarrier over all `N` amplitude samples in the clip):

- Mean: `μ = (1/N) Σ xᵢ`
- Standard deviation (population, not sample): `σ = sqrt( (1/N) Σ (xᵢ − μ)² )`
- **Skewness** (Fisher–Pearson coefficient of skewness, no bias correction): `skew = (1/N) Σ (xᵢ − μ)³ / (σ³ + ε)` — measures asymmetry of the amplitude distribution within the clip (e.g. a person's body creating a lopsided fade pattern vs. a symmetric one)
- **Excess kurtosis** (Fisher's definition — 0 for a Gaussian): `kurt = (1/N) Σ (xᵢ − μ)⁴ / (σ⁴ + ε) − 3` — measures how peaked/heavy-tailed the amplitude distribution is (motion tends to produce heavier tails than standing still)
- `ε = 1e-6` guards division by zero for subcarriers that are constant across the whole clip (e.g. zero-padded leading subcarriers); the resulting NaN is then mapped to 0.

Together `[mean, std, skew, kurt]` are the first four **statistical moments** — a classic,
model-agnostic way to summarize a distribution's shape without assuming any particular
parametric form (no assumption that amplitude is Gaussian, etc.).

**Feature vector**: `4 × n_sub` floats per clip — `[mean(n_sub), std(n_sub), skew(n_sub), kurt(n_sub)]` concatenated.

**Cross-width padding (v1 only)**: v1 pools Day1 (186 sub, HT40) + Day2 (128 sub, HT20). Day2's
vector is padded `4×128 → 4×186` by zero-filling each of the 4 moment blocks *independently*
(`stack_features`) rather than naively concatenating zeros at the end — this keeps each
subcarrier's moment block internally contiguous so the classifier still sees "mean block, std
block, skew block, kurt block" in a consistent layout regardless of width.

No empty-room (`calibA`) z-scoring is applied to the deployed artifact's own real dimensions.
**Why that's safe here**: Random Forest splits a feature at a threshold (`feature > c`); shifting
and scaling a feature (`x' = (x − μ)/σ`, standard **z-score standardization**) is a monotonic
affine transform, so the optimal split threshold just moves by the same affine transform — the
tree's decisions are provably unchanged. Z-scoring is only needed for the *cross-day padding*
variant, where it gives a principled zero-fill value ("at empty-room baseline") for the missing
dimensions, not because RF needs it for its own real features.

### 1.2 Feature dimensionality

| Version | Native subcarriers | Feature vector length | Why |
|---|---|---|---|
| v1 | 128 (Day2) zero-padded to 186 (Day1) | `4 × 186 = 744` | Day1 = 186-sub/HT40, Day2 = 128-sub/HT20 — different capture modes, so v1 pads to the larger width |
| **v2 (deployed default)** | 128 (both days native) | `4 × 128 = 512` | Day3 (2026-09-15) and Day4 (2026-09-16) are **both** native 128-subcarrier / channel 6 / 20 MHz — no cross-mode padding needed, `N_TARGET = 128` |

### 1.3 Model — Random Forest

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

**How Random Forest works** (Breiman, "Random Forests," *Machine Learning*, 2001) — an ensemble
of decision trees combining two sources of randomness:

1. **Bagging (bootstrap aggregating)**: each of the 300 trees is trained on an independent
   bootstrap resample (sampling `N` rows with replacement from the `N` training clips) — some
   clips appear multiple times, some not at all, per tree.
2. **Random feature subsampling**: at each split in each tree, only a random subset of the 512
   (or 744) features is considered (scikit-learn's classification default is `sqrt(n_features)`
   candidate features per split) — decorrelates the trees so they don't all latch onto the same
   dominant subcarrier.
3. **Split criterion**: each candidate split `feature > threshold` is scored by **Gini impurity**
   reduction — for a node with class proportions `pₖ`, `Gini = 1 − Σ pₖ²`; the split that
   maximizes the *decrease* in weighted child-node Gini impurity is chosen.
4. **Prediction**: majority vote (or averaged class probability) across all 300 trees'
   predictions for a given input.

`class_weight="balanced"` reweights each class inversely proportional to its frequency
(`n_samples / (n_classes × n_samples_of_that_class)`) during the Gini-impurity calculation, so
the model isn't biased toward whichever person happened to contribute more clips (see the
uneven per-day clip counts in §1.4).

**Why Random Forest over a deep model**: stated directly in the training script's docstring —
across every cross-day comparison run in this project (crop-128, Variant-B, paper-recipe+Variant-A,
clip-recipe), RF matched or beat every deep-model variant tried and never fully collapsed to a
single class the way the transformer did on several runs. This tracks with general ML folklore:
tree ensembles tend to be more robust than deep nets on small, noisy, tabular-style data (a few
thousand hand-engineered feature vectors, not millions of raw sequences) because they don't need
to learn a representation from scratch.

Trained on **100% of the pooled training data** (no internal held-out split) — the "train-set
fit accuracy" printed (100.0% for both v1 and v2) is a sanity check only, not a generalization
estimate; it just confirms the trees can memorize their own bootstrap samples, which is expected
of any sufficiently deep, unpruned tree ensemble.

### 1.4 Training data

| | v1 | v2 (deployed default) |
|---|---|---|
| Dates | 2026-09-09, 2026-09-10 (Day1+Day2) | 2026-09-15, 2026-09-16 (Day3+Day4) |
| Total clips | 4,316 | **6,516** |
| Per day/person | anjali/barath ≈ 1,000–1,170 each | anjali 1,854 (D3) + 1,174 (D4); barath 1,615 (D3) + 1,873 (D4) |
| Motion | both pooled (standing + walking) | both pooled |
| Cross-day validated before shipping? | Yes (Day1→Day2 proxy) | **No** — Day3→Day4 was never itself cross-day-tested before v2 shipped; v2 just pools both days |

### 1.5 Postprocessing / inference (`score_live_session.py`)

1. Same clip recipe (3 s / 1 s stride / 50% coverage) is applied to the new session, using the
   artifact's stored `n_target` (128 for v2) — clips with `n_sub != n_target` get the same
   zero-pad-per-block treatment as training.
2. `model.predict_proba` → per-clip `predicted` label + `confidence` (the fraction of the 300
   trees that voted for that class).
3. **Majority vote across all clips in the session** is the trusted session-level output.
   **Why**: each clip is a noisy, independent 3 s sample and clips overlap by 2/3 (1 s stride vs
   3 s length), so consecutive predictions are correlated but individually noisy — the
   session-level vote acts like averaging many weak, partially-independent estimates, which
   reduces variance the same way any ensemble/voting scheme does.
4. If `--true-person` is given: per-clip accuracy and session-level (majority-vote) correctness
   are reported against it.
5. **No reject option.** Closed 2-class problem — a genuine stranger is still forced into
   `anjali` or `barath`; there is no calibration/empty-room baseline step in this pipeline at
   all (unlike `live_infer.py`'s `calibA`).

### 1.6 Measured accuracy (stated plainly in the runbook/docstrings)

- Same-day: ~70–95% (v2 runbook) / ~70–80% (train script docstring), varies a lot by motion.
- Cross-day: ~43–69% (Day1+2→Day3), ~55–70% general range quoted in the v2 runbook.
- Day3→Day4 itself: never isolated/tested — v2 pools rather than validates.

---

## 2. Signature Model + Extreme Value Machine (open-set identification)

Source: `ml/models/signature_evm.py` (model + EVM math), trained by
`ml/training/train_signature_evm_day3day4.py`. Ported from the sibling
`WiFi-Sense/csi_pipeline` project's WhoFi-style signature model + EVM open-set decision layer.
Unlike the home model, this one is designed to say **"unknown,"** not just rank two known
classes — that's the entire reason it exists (see the false-accept problem in §2.4).

### 2.1 Preprocessing pipeline

| Step | What | How | Why |
|---|---|---|---|
| Session pool | Day3+Day4, channel-6-only | Same manifest-builder as `train_day3_day4_ch6_model.py` (`build_day3_day4_ch6_manifest`) | Reuses an already-fixed packet-rate confound (see project notes on the Day3-channel-6 fix) instead of re-deriving it |
| Time normalization | Resample to a common packet rate | `mode="resampled_timenorm"`: bin-average packets to a target rate derived from the pooled Day3+Day4 sessions | Different sessions/days have slightly different native packet rates; without this, a model could learn to distinguish sessions by packet *density* instead of by CSI content |
| Windowing | Fixed-length sequence input for the transformer | `window_packets=200`, `stride_packets=100` (50% overlap) | Unlike the home model's per-clip moment reduction, a transformer *wants* the raw sequence (it has attention to summarize sequences itself), so windows are packet counts, not a moment vector |
| Calibration | None | Raw amplitude, no empty-room z-scoring | Deliberate: the reference project explicitly tested calibA for this contrastive setup and found it neutral-to-harmful — it was answering "is this data authorized" implicitly via the calibration statistics, not letting the encoder learn identity signal itself |
| Channel used | Amplitude only, no phase | `amplitude = sqrt(real² + imag²)` per subcarrier, same as §1.1 | Matches WhoFi's own published design choice |
| Identities trained on | All identities in the pool, not just the 2 enrolled | anjali, barath **+** abdul, divya, harshitha, kishore, manas, sumanth | An open-set rejection boundary is meaningless without impostor examples to define what "not enrolled" looks like — training only on anjali/barath would give the EVM nothing to fit a tail against |

Input shape to the encoder: `(B, T=200, n_sub=128)`.

### 2.2 Model architecture — `SignatureModel`

```
amplitude (B, T=200, n_sub=128)
      │
      ▼
BranchEncoder (ml/models/common.py)
  ├─ Linear input projection: 128 → d_model=32
  ├─ Sinusoidal positional encoding (max_len=1024)
  └─ TransformerEncoder × num_layers=1
        - nhead = 4
        - dim_feedforward = 64
        - dropout = 0.2
        - post-LN (norm_first=False)
      │  output: (B, T, 32)
      ▼
mean-pool over time  →  (B, 32)
      │
      ▼
SignatureHead
  ├─ Linear: 32 → signature_dim=32
  └─ L2-normalize (unit hypersphere)
      │
      ▼
signature embedding (B, 32), ‖s‖=1
```

**How the Transformer encoder works** (**scaled dot-product multi-head self-attention**,
Vaswani et al., *"Attention Is All You Need,"* NeurIPS 2017):

- **Sinusoidal positional encoding** injects order information since attention itself is
  permutation-invariant:
  `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`,
  added elementwise to the projected input.
- **Scaled dot-product attention**, per head: `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`, where
  `Q, K, V` are linear projections of the input and `d_k` is the per-head dimension
  (`d_model / n_heads = 32/4 = 8` here). Multi-head attention runs this in parallel across 4
  heads and concatenates the results, letting different heads attend to different subcarrier
  relationships simultaneously.
- Each encoder layer is the standard **post-LN Transformer block**: `x = LayerNorm(x + Dropout(SelfAttention(x)))`, then `x = LayerNorm(x + Dropout(FFN(x)))`, where `FFN(x) = Linear(ReLU(Linear(x)))` expands to `d_ff=64` and back.
- **Mean-pooling** over the time dimension collapses the `(B, T, 32)` token sequence to one
  `(B, 32)` vector per window — a simple, parameter-free way to aggregate a variable-length
  window into a fixed-size representation (an alternative to a `[CLS]` token or attention
  pooling, not used here).
- **L2 normalization** (`x / ‖x‖₂`) projects the embedding onto the unit hypersphere, so
  Euclidean distance and cosine similarity between two embeddings become monotonically related
  (`‖a−b‖² = 2 − 2·cos(a,b)` for unit vectors) — this is what makes the EVM's Euclidean-distance
  math in §2.4 equivalent to a similarity/angle comparison.

`d_model=32`, 1 layer, 4 heads is **deliberately small** — the module docstring cites both the
WhoFi paper's and the ESP32 person-ID paper's finding that going deeper hurt on this class of
small CSI dataset (a few thousand windows, not the million-scale data transformers are usually
tuned for), i.e. classic overfitting risk from an over-parameterized model on a small dataset.

### 2.3 Training objective — in-batch negative contrastive loss

**Named method**: this is the **DPR-style in-batch negatives loss** (Karpukhin et al., *"Dense
Passage Retrieval for Open-Domain Question Answering,"* EMNLP 2020), a close relative of the
broader **InfoNCE / contrastive loss** family (van den Oord et al. 2018; SimCLR, CLIP, etc.).

**How**:
1. **PK-sampling** (a batching scheme from person re-identification literature, e.g. Hermans,
   Beyer & Leibe, *"In Defense of the Triplet Loss for Person Re-Identification,"* 2017): every
   identity with ≥2 windows contributes exactly one `(query, gallery)` pair per step, aligned by
   identity index — this guarantees every batch has at least one genuine positive pair per
   identity, which random i.i.d. sampling wouldn't reliably do with only 8 identities.
2. **Context-aware pair selection**: query and gallery are drawn from *different* `(date, motion)`
   contexts when available. **Why**: a diagnostic (`_diagnose_signature_evm.py`) found the
   encoder was otherwise learning to separate recording-session context rather than identity —
   the closest cross-person pair in embedding space was two *different* strangers, same day,
   same motion (distance 0.068), closer than typical same-identity pairs. Forcing cross-context
   positives makes the contrastive loss only satisfiable by identity signal that survives a
   context change.
3. **Loss math**: with `Sq, Sg ∈ ℝ^(B×32)` the L2-normalized query/gallery signature batches,
   `sim = Sq · Sgᵀ ∈ ℝ^(B×B)` is a cosine-similarity matrix (dot product of unit vectors =
   cosine of the angle between them). Cross-entropy is applied row-wise with the diagonal as the
   target class:
   `L = −(1/B) Σᵢ log( exp(simᵢᵢ) / Σⱼ exp(simᵢⱼ) )`
   — i.e. a softmax over each row's similarities to every gallery embedding in the batch,
   pushing the true match (diagonal) toward 1 and every other identity in the batch (in-batch
   negatives — free negatives, no separate negative-mining step needed) down.

**Hyperparameters:**

| Param | Value |
|---|---|
| `SIGNATURE_DIM` | 32 |
| `D_MODEL` | 32 |
| `EPOCHS` | 300 (40 undertrained the encoder: loss barely moved off `ln(8)≈2.05 → 1.88`, the theoretical random-chance cross-entropy for an 8-identity batch, and produced AUROC 0.436, worse than chance) |
| `STEPS_PER_EPOCH` | 40 |
| Optimizer | Adam (Kingma & Ba, 2014), `lr=1e-3` |
| Seed | 0 |

### 2.4 Extreme Value Machine (the accept/reject decision layer)

**Why EVM instead of a closed-set classifier**: a plain softmax classifier (the `taskD`
checkpoint) measured a **68.5% mean false-accept rate** against a genuinely novel stranger — by
construction, a closed-set classifier's output is a probability distribution *over its known
classes*, so it structurally cannot express "none of these." Nearest-centroid rejection
(accept if within some distance of the nearest enrolled centroid) scored 0% genuine accept at
any usable threshold on held-out data. The **Extreme Value Machine** (Rudd, Jain, Scheirer &
Boult, *"The Extreme Value Machine,"* IEEE TPAMI) was the first approach in the reference
project that produced a usable, threshold-able open-set decision.

**The statistical idea (Extreme Value Theory)**: EVM's premise is that the *distance from a
point to its nearest points of another class* behaves, in the limit, like an extreme-value
(minimum) statistic — analogous to how the **Fisher–Tippett–Gnedenko theorem** says the
distribution of block-minima of i.i.d. samples converges to one of a small family of extreme-value
distributions (here, modeled as Weibull, appropriate for a bounded-below quantity like a
distance). Instead of assuming a single global decision boundary, EVM fits a **per-point Weibull
tail** to each support point's nearest-`tau` distances to *other* classes, giving each point its
own local notion of "how far is far enough to call something else."

**Fitting (`fit_evm`)** — ported from the reference project's published `tau20` config:

| Param | Value | Notes |
|---|---|---|
| `tau` | 20 | nearest-OTHER-identity distances per point used to fit the Weibull tail |
| `equalize` | True | caps every identity to the smallest identity's point count (skipped in centroid mode), so no identity's EVM is denser purely from having more training windows |
| `centroid_only` | **True** (this run) | collapses each identity's "own" points to a single centroid before fitting — see below |
| Fit | `scipy.stats.weibull_min.fit(near, floc=0)` → `(shape k, scale λ)` per support point, via maximum-likelihood estimation with the location fixed at 0 |
| Guardrails | `scale` capped at `MAX_SANE_SCALE=8.0` (4× the max possible Euclidean distance between two unit vectors, 2.0) and `shape` capped at `MAX_SANE_SHAPE=20.0` (a diagnostic found degenerate fits blowing shape up to 82.4, which turns the score function below into a near step function); fits outside range fall back to `(k=1.0, λ=mean(near))` |

**Why `centroid_only=True`**: per-point EVM (~289 individual Weibull fits per identity after
equalize-capping) scored mean AUROC **0.408–0.466**, noisier than a plain nearest-centroid rule
on the *same embeddings* (AUROC **0.737**). With only ~289 points per identity, hundreds of
individual tail fits are too high-variance; collapsing "own" to one centroid per identity trades
per-point granularity for one much more stable Weibull tail, at the cost of losing intra-identity
shape information.

**Scoring (`evm_score`) — the probability-of-sample-inclusion (ψ) function**, which is exactly
the **survival function (1 − CDF) of the fitted Weibull distribution**, evaluated at the
distance from a new point `x` to a support point:

```
ψ(x, point) = exp( −( ‖x − point‖ / λ )^k )        (Weibull CDF: F(d) = 1 − exp(−(d/λ)^k))
```

- `ψ = 1` at zero distance, decaying toward 0 as `x` moves further from the support point — it's
  literally "the probability that `x` is *not* an extreme (too-far) outlier relative to this
  point," used here as an inclusion/similarity score rather than its usual outlier-detection role.
- **Deliberate change from the reference project**: instead of `max()` over an identity's support
  points (found to have zero robustness — one degenerate point can single-handedly make an
  identity accept almost anything, since `max()` only needs one point to agree), this averages
  the **top-5** (`top_k=5`) highest ψ values per identity — a compromise between "any one close
  point is enough" (`max`, fragile) and "every point must agree" (`mean` over all points, too
  strict), rewarding a genuine close match without being dominated by one outlier.
- Final accept score per test embedding = `max` over enrolled identities' ψ; predicted identity
  = `argmax`.

**Decision threshold**: `ACCEPT_THRESHOLD = 0.18` — centroid-only mode's ψ scores run much
lower than per-point mode's (only 1 support point vs. ~289, so top-5-averaging effectively
reduces to a single point's ψ), so the "obvious" 0.5 threshold accepted/rejected nothing in
every fold. 0.18 was picked as the empirical **EER (Equal Error Rate)** — the threshold where
false-accept rate equals false-reject rate — on the 3 strangers the model handles reasonably
(harshitha/kishore/manas: EER 9–22%); abdul/divya/sumanth stay near-chance regardless of
threshold.

### 2.5 Evaluation protocol & honest numbers

- **Leave-one-unauthorized-person-out** cross-validation: for each held-out stranger, refit the
  EVM *without* them, then check whether their windows get rejected and whether held-out
  enrolled windows get accepted **as the correct identity** (not just "some enrolled identity").
  This is the open-set analog of leave-one-out CV, specifically checking generalization to a
  truly novel impostor rather than one seen during EVM fitting.
- `assert_no_group_leakage` on `session_dir` enforced per fold — no window from the same
  recording session appears in both train and test, which would otherwise let the model exploit
  session-specific artifacts instead of identity.
- Latest run (2026-09-17T07:26) mean across 6 held-out strangers:
  **AUROC ≈ 0.446, genuine-accept ≈ 0.171, correct-identity-given-accepted ≈ 0.558,
  false-accept-on-stranger ≈ 0.322**.
- These numbers vary run-to-run (see full history in
  `ml/evaluation/results/signature_evm_day3day4_log.csv`) and are explicitly called out in the
  training script as "the trustworthy numbers, not the final EVM's" — the deployed checkpoint's
  EVM is refit on 100% of the data afterward and was **not** re-validated against these folds.

### 2.6 Deployed artifact

`ml/checkpoints/signature_evm_day3day4.pt` — contains:

- `model_state` (SignatureModel weights)
- `arch_kwargs`: `n_subcarriers=128, d_model=32, signature_dim=32`
- `evm`: fitted Weibull params for the 2 **enrolled** identities only (anjali, barath), fit on
  100% of the pooled Day3+Day4 data
- `evm_tau=20`, `accept_threshold=0.18`
- `meta`: train dates, `target_channel=6`, `target_rate_hz`, `window_packets=200`,
  `stride_packets=100`, `calibration="none"`, seed, creation timestamp

### 2.7 Explicitly NOT implemented (documented next levers, not bugs)

- **Cohort / T-norm score normalization** — a speaker/face-verification technique that
  normalizes a raw score against a cohort of impostor scores per trial; the reference project's
  single biggest false-accept-rate reducer, not ported here.
- Per-identity EER thresholds (one global threshold of 0.18 is used instead)
- Multi-seed ensembling
- Walking-motion gating
- Temporal smoothing over consecutive windows
- Any cross-day validation beyond Day3+Day4 (no Day5 exists yet)

---

## 3. Known / named methods referenced above

| Method | Where it's used | Reference |
|---|---|---|
| Bagging (bootstrap aggregating) + Random Forest | Home model | Breiman, "Random Forests," *Machine Learning*, 2001 |
| Gini impurity split criterion | Random Forest's internal tree-building | Standard CART algorithm (Breiman et al., 1984) |
| Fisher–Pearson skewness / Fisher excess kurtosis | Home model's per-clip features | Classical method-of-moments statistics |
| Z-score standardization | Cross-day feature padding baseline | Standard statistical normalization |
| Scaled dot-product multi-head self-attention / Transformer encoder | `BranchEncoder` in both the signature model and the WhoFi transformer used elsewhere in this repo | Vaswani et al., "Attention Is All You Need," NeurIPS 2017 |
| Sinusoidal positional encoding | Same `BranchEncoder` | Vaswani et al., 2017 (same paper) |
| PK-sampling (P identities × K samples/batch) | Signature model's batch construction | Hermans, Beyer & Leibe, "In Defense of the Triplet Loss for Person Re-Identification," 2017 |
| In-batch-negatives contrastive loss (DPR-style) | Signature model's training loss | Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering," EMNLP 2020; broader family: InfoNCE (van den Oord et al., 2018), SimCLR/CLIP |
| L2-normalized embeddings on the unit hypersphere / cosine-similarity metric learning | Signature model's output head | Common practice across face/speaker/person re-id embedding models |
| Extreme Value Theory / Fisher–Tippett–Gnedenko theorem | Statistical basis for EVM's per-point Weibull tail | Classical extreme value statistics |
| Extreme Value Machine (EVM) | Open-set accept/reject decision layer | Rudd, Jain, Scheirer & Boult, "The Extreme Value Machine," IEEE TPAMI |
| Weibull distribution (MLE fit, survival function as inclusion score) | EVM's tail model | `scipy.stats.weibull_min` |
| Equal Error Rate (EER) | EVM's accept-threshold selection | Standard biometric-verification metric |
| WhoFi (transformer + amplitude-only CSI + in-batch-negative contrastive loss for WiFi-based person re-identification) | Overall architecture pattern for the signature model | Referenced throughout this codebase's docstrings as "the WhoFi paper" |

---

## Quick comparison

| | Home Model (v2) | Signature + EVM |
|---|---|---|
| Task | Closed 2-class ID (anjali/barath) | Open-set accept/reject + ID |
| Model | RandomForest (300 trees) | Transformer encoder (32-d) + Weibull/EVM |
| Feature dim | 512 (4×128 per-subcarrier moments) | 128-sub raw amplitude → 32-d embedding |
| Windowing | 3 s clips / 1 s stride, 50% coverage gate | 200-packet windows / 100-packet stride |
| Calibration | None (RF invariant to affine transform) | None (found harmful for this contrastive setup) |
| Can say "unknown"? | No | Yes (that's the point of EVM) |
| Training data | Day3+Day4, anjali+barath only | Day3+Day4, anjali+barath **+ 6 strangers** |
