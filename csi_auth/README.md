# csi_auth -- from-scratch CSI identity/auth pipeline

A second, independently-written pipeline for the Anjali/Barath person-identification and
auth-vs-non-auth questions -- built without importing any of `ml/`'s code, as a cross-check against
the `ml/` + `analysis/*_ch6` pipeline elsewhere in this repo. Reads `data/` directly (walking motion,
channel-6/20MHz sessions only, one fixed receiver board, detected per-session from the raw packet
fields rather than assumed from dates).

## Pipeline

`data.py` (scan sessions) -> `decode.py` (raw CSI -> amplitude/phase) -> `cleaning.py` (Hampel-filter
amplitude spikes) -> `subcarrier_mask.py` (drop 19 verified-null guard/DC/pilot bins) ->
`windowing.py` + `features.py` (handcrafted stats) -> `dataset.py` (caches the window-level feature
matrix + raw amplitude sequences) -> `models.py` (SVM, RandomForest, HistGradientBoosting,
CNN+BiLSTM, CNN+self-attention) -> `train.py` / `identity.py` (leave-one-day-out evaluation, both
window-level and session-aggregated) -> `permutation_test.py` / `identity_permutation.py`
(session-label-shuffle chance-floor control) -> `demo_timing.py` (live-demo question: accuracy as a
function of how many chronological windows/seconds of walking have been observed so far).

`cache/*.csv` holds the actual result rows from the runs described below (small, checked in). The
large derived arrays (`X_stats.npy`, `X_seq.npy`, decoded per-session caches) are NOT checked in
(gitignored) -- rerun `python dataset.py` to regenerate them from `data/`.

## Headline results (leave-one-day-out, 6 channel-6 days, `data/`)

**Anjali vs Barath (closed 2-person identity), walking:**
- Window-level (~1 window ≈ a few hundred ms - few seconds, varies with capture rate): ~66-74%
  accuracy depending on model, all confirmed 3.6-4.6 std above a session-label-shuffle chance floor
  (~44-53%) -- see `identity_permutation.py` output.
- **Session-level (average a model's confidence across a whole test session before deciding)**:
  80-90% accuracy. CNN+BiLSTM highest at 90.0%. Confirmed NOT an artifact of aggregating noise --
  the same shuffled-label control run through session-aggregation stays flat at ~44-53%, while real
  labels jump ~37-42 points above it.
- **Live-demo timing** (`demo_timing.py`, chronological/sequential windows -- no peeking ahead, unlike
  the random-subsampled session-level number above): accuracy climbs from ~65-82% at 10 windows
  (~4s) to ~84-90% by 150 windows (median ~60s of continuous walking), matching the session-level
  ceiling. CNN+Attention is the strongest model at nearly every checkpoint.

**Auth (Anjali/Barath) vs non-auth (any other person)**: NOT reliably above chance (~52-54% vs a
~49-50% shuffled floor, well within 1-1.6 std) -- neither spike-cleaning nor subcarrier-masking
changed this. Most likely cause: the non-auth class is data-starved (~18 different strangers, mostly
one session each), not a preprocessing problem. See conversation history / commit history for the
full diagnostic trail (packet-rate check, null-subcarrier verification, permutation controls).

## Relationship to `analysis/standing_ch6` and `analysis/walking_ch6`

Those two directories are a separate, earlier pipeline (built on top of `ml/data_pipeline`) answering
the same standing-vs-walking identifiability question. The two pipelines' conclusions agree on the
core finding (walking carries a real, day-stable Anjali-vs-Barath signal; standing's is weaker and
more session/day-confounded) -- see each directory's `FINDINGS.md`.
