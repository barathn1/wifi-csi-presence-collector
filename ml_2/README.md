# ml_2 -- channel-6-only, GPU-trained open-set model battery

See `PLAN.md` for the full reasoning. Quick reference for running things.

Want to run `gnn_subcarrier`/`radar_cnn` live against the ESP32 instead of offline? See
`LIVE_TEST_RUNBOOK.md`.

## Setup

Everything here imports `ml.*` directly, so run from the repo root, same convention as `ml/`:

```bash
python3 -m ml_2.training.train_svm_gait
```

`torch` needs CUDA support (reinstalled from the plain PyPI wheel during this round -- was a
`+cpu`-only build before). `xgboost` is required for the GBM baseline. Both installed at the
user/site level (`pip install --break-system-packages ...`), not an isolated venv -- `python3.14-venv`
isn't installed on this box and there's no passwordless sudo to add it.

## Run everything

```bash
python3 -m ml_2.training.run_all       # fires every model in sequence, then the consolidated backtest
```

Or one model at a time (each is independently runnable):

```bash
python3 -m ml_2.training.train_svm_gait
python3 -m ml_2.training.train_gbm
python3 -m ml_2.training.train_prototypical --epochs 8
python3 -m ml_2.training.train_arcface --epochs 8
python3 -m ml_2.training.train_transformer_gpu --epochs 6
python3 -m ml_2.training.train_receiver_fusion_gpu --epochs 6
python3 -m ml_2.evaluation.backtest_all   # combine whichever logs already exist
```

## Where results land

- `ml_2/evaluation/results/{svm_gait,gbm,prototypical,arcface,transformer_gpu,receiver_fusion_gpu}_log.csv`
  -- one row per fold, per model, per split type.
- `ml_2/evaluation/results/backtest_summary.csv` -- the consolidated mean-across-folds comparison
  table, grouped by (model, split_type). Printed to stdout too, split into the 3 tiers that matter:
  **open-set** (leave-one-unauthorized-person-out -- the real stranger-rejection number),
  **cross-day** (leave-one-day-out -- the real generalization-to-a-new-day number), and
  **closed-set sanity** (session-disjoint 5-fold -- the easy number, reported for context only).

## Reading the numbers

- `auroc`/`eer` are the metrics to compare models on, not raw `accuracy` -- `ml.evaluation.metrics`'s
  own docstring notes EER should stay under 5% for security-grade use per the CSI-biometrics SoK survey.
- `false_accept_unauthorized` / `false_accept_none` matter more than blended accuracy: a model that
  never lets a real stranger in but occasionally misses an authorized person is a very different
  outcome from one that's accurate on average but lets strangers through.
- The **open-set** and **cross-day** numbers are the honest ones. The **closed-set sanity** number will
  look better than either -- that gap IS the finding, not noise to explain away.
