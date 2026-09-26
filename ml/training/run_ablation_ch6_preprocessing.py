"""Cross-day preprocessing ablation study for channel-6 person identification (2026-09-15/16/17).

Same model architectures, same 200-packet sequence length, same leave-one-day-out cross-day split,
and same task (`taskF_identity_or_nonauth`: anjali / barath / non_auth) across all 8 experiments --
ONLY the preprocessing changes:

    E1 raw amplitude                         E5 hampel + smoothing
    E2 hampel                                E6 moving_iqr + smoothing
    E3 iqr                                   E7 hampel + smoothing + normalization
    E4 moving_iqr                            E8 moving_iqr + smoothing + normalization
    E9 hampel + butterworth (paper exact)    E10 E9 + normalization

E9/E10 replicate arXiv:2507.12854's exact noise-reduction recipe (Hampel window=15/n_sigmas=3 +
5th-order Butterworth low-pass cutoff=10Hz) instead of E5-E8's moving-average smoothing, so this
harness can directly answer "does the paper's own preprocessing choice generalize cross-day on our
channel-6 data" on the same task/split/metrics as every other experiment here.

Normalization statistics are computed ONLY from each fold's training dates
(`ablation_ch6_pipeline.compute_train_normalization_stats`) and reused unchanged on the held-out
day -- never recomputed on, or informed by, the test day.

Two models share every config: the existing amplitude-only BiLSTM baseline
(`ml.models.lstm.BiLSTMAmplitudeOnly`) and a Transformer (`ml.models.transformer_whofi.WhoFiTransformer`,
called amplitude-only). Every (experiment, model, held-out day) cell is averaged over `--seeds` seeds
(this project's own convention -- see `ml/training/train.py`'s seed-variance note).

The primary objective is CROSS-DAY performance, not same-day accuracy: every reported number is
from a held-out day never seen during that fold's training.

    python3 -m ml.training.run_ablation_ch6_preprocessing
    python3 -m ml.training.run_ablation_ch6_preprocessing --epochs 6 --seeds 0 1 2 --experiments E1_raw E7_hampel_smooth_norm
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data_pipeline.ablation_ch6_pipeline import (
    TARGET_DATES,
    build_ch6_manifest_all_labels,
    build_or_load_window_index,
    compute_dataset_target_rate_hz,
    compute_train_normalization_stats,
    make_normalizer,
)
from ml.data_pipeline.ablation_preprocessing import EXPERIMENTS, PreprocessConfig
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.evaluation.metrics import compute_auroc
from ml.models.lstm import BiLSTMAmplitudeOnly
from ml.models.tfmamba_approx import TFMambaApprox
from ml.models.transformer_whofi import WhoFiTransformer

CLASSES = ["anjali", "barath", "non_auth"]
NON_AUTH_IDX = CLASSES.index("non_auth")
MODELS = ["bilstm", "transformer", "mamba"]
LOG_PATH = REPO_ROOT / "ml/evaluation/results/ablation_ch6_preprocessing_log.csv"
LOG_FIELDS = [
    "timestamp", "experiment", "model", "held_out_date", "accuracy_mean", "accuracy_std",
    "auroc_mean", "auroc_std", "far_unauthorized_mean", "far_none_mean",
    "recall_anjali_mean", "recall_barath_mean", "n_train", "n_test", "epochs", "seeds", "notes",
]


def build_model(model_name: str) -> nn.Module:
    if model_name == "bilstm":
        return BiLSTMAmplitudeOnly(n_subcarriers=128, n_classes=len(CLASSES))
    if model_name == "transformer":
        return WhoFiTransformer(n_subcarriers=128, n_classes=len(CLASSES))
    if model_name == "mamba":
        return TFMambaApprox(n_subcarriers=128, n_classes=len(CLASSES))
    raise ValueError(model_name)


def train_and_predict(model: nn.Module, train_ds: CsiWindowDataset, test_ds: CsiWindowDataset,
                       epochs: int, seed: int, batch_size: int = 64, lr: float = 1e-3):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    for _ in range(epochs):
        model.train()
        for amp, _phase, label in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(amp), label)
            loss.backward()
            opt.step()

    model.eval()
    probs_list, true_list = [], []
    with torch.no_grad():
        for amp, _phase, label in test_loader:
            logits = model(amp)
            probs_list.append(torch.softmax(logits, dim=-1).numpy())
            true_list.append(label.numpy())
    probs = np.concatenate(probs_list)
    true = np.concatenate(true_list)
    return probs, true


def compute_fold_metrics(probs: np.ndarray, true: np.ndarray, raw_label: np.ndarray) -> dict:
    """raw_label: the ORIGINAL per-window label (authorized/unauthorized/none), aligned row-for-row
    with `true`/`probs` (test_ds's DataLoader uses shuffle=False, so this is safe) -- lets FAR-
    unauthorized and FAR-none be told apart even though taskF trained on them as one `non_auth` class."""
    pred = probs.argmax(axis=-1)
    accuracy = float((pred == true).mean())

    auth_score = probs[:, CLASSES.index("anjali")] + probs[:, CLASSES.index("barath")]
    y_is_auth = (true != NON_AUTH_IDX).astype(int)
    auroc = compute_auroc(y_is_auth, auth_score)

    pred_is_auth = pred != NON_AUTH_IDX
    mask_unauth = raw_label == "unauthorized"
    far_unauthorized = float(pred_is_auth[mask_unauth].mean()) if mask_unauth.any() else float("nan")
    mask_none = raw_label == "none"
    far_none = float(pred_is_auth[mask_none].mean()) if mask_none.any() else float("nan")

    def _recall(person_idx: int) -> float:
        mask = true == person_idx
        return float((pred[mask] == person_idx).mean()) if mask.any() else float("nan")

    return {
        "accuracy": accuracy, "auroc": auroc, "far_unauthorized": far_unauthorized, "far_none": far_none,
        "recall_anjali": _recall(CLASSES.index("anjali")), "recall_barath": _recall(CLASSES.index("barath")),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def run_experiment(config: PreprocessConfig, manifest: pd.DataFrame, epochs: int, seeds: list[int],
                    models: list[str] = MODELS) -> list[dict]:
    print(f"\n=== {config.name} ===")
    target_rate_hz = compute_dataset_target_rate_hz(manifest, config)
    window_index = build_or_load_window_index(manifest, config, target_rate_hz)

    rows = []
    for held_out in TARGET_DATES:
        train_dates = [d for d in TARGET_DATES if d != held_out]
        train_index = window_index[window_index["date"].isin(train_dates)].reset_index(drop=True)
        test_index = window_index[window_index["date"] == held_out].reset_index(drop=True)
        if train_index.empty or test_index.empty:
            continue

        normalizer = None
        if config.normalize:
            mean, std = compute_train_normalization_stats(train_index)
            normalizer = make_normalizer(mean, std)

        train_ds = CsiWindowDataset(train_index, "taskF_identity_or_nonauth", calibration=normalizer, classes=CLASSES)
        test_ds = CsiWindowDataset(test_index, "taskF_identity_or_nonauth", calibration=normalizer, classes=CLASSES)
        raw_label = test_ds.index["label"].values

        for model_name in models:
            per_seed = []
            for seed in seeds:
                model = build_model(model_name)
                probs, true = train_and_predict(model, train_ds, test_ds, epochs, seed)
                per_seed.append(compute_fold_metrics(probs, true, raw_label))

            agg = {}
            for key in per_seed[0]:
                mean_v, std_v = _mean_std([m[key] for m in per_seed])
                agg[f"{key}_mean"], agg[f"{key}_std"] = mean_v, std_v

            print(f"  {model_name:<12} held_out={held_out} train={train_dates} "
                  f"acc={agg['accuracy_mean']*100:.1f}% auroc={agg['auroc_mean']:.3f} "
                  f"far_unauth={agg['far_unauthorized_mean']*100:.1f}% far_none={agg['far_none_mean']*100:.1f}% "
                  f"recall_anjali={agg['recall_anjali_mean']*100:.1f}% recall_barath={agg['recall_barath_mean']*100:.1f}%")

            rows.append({
                "timestamp": datetime.now(timezone.utc).isoformat(), "experiment": config.name, "model": model_name,
                "held_out_date": held_out, "accuracy_mean": agg["accuracy_mean"], "accuracy_std": agg["accuracy_std"],
                "auroc_mean": agg["auroc_mean"], "auroc_std": agg["auroc_std"],
                "far_unauthorized_mean": agg["far_unauthorized_mean"], "far_none_mean": agg["far_none_mean"],
                "recall_anjali_mean": agg["recall_anjali_mean"], "recall_barath_mean": agg["recall_barath_mean"],
                "n_train": len(train_ds), "n_test": len(test_ds), "epochs": epochs, "seeds": str(seeds),
                "notes": f"train={train_dates}",
            })
    return rows


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=LOG_FIELDS)
    df.to_csv(LOG_PATH, mode="a" if LOG_PATH.exists() else "w", header=not LOG_PATH.exists(), index=False)


def main(experiment_names: list[str], epochs: int, seeds: list[int], models: list[str] = MODELS) -> None:
    manifest = build_ch6_manifest_all_labels()
    print(f"channel-6 sessions across {TARGET_DATES}: {len(manifest)}")
    print(manifest.groupby(["date", "label"]).size())

    all_rows = []
    for name in experiment_names:
        all_rows += run_experiment(EXPERIMENTS[name], manifest, epochs, seeds, models=models)
    log_rows(all_rows)
    print(f"\nlogged {len(all_rows)} rows to {LOG_PATH}")

    print("\n=== CROSS-DAY SUMMARY (mean over the 3 held-out days) ===")
    summary = pd.DataFrame(all_rows)
    for model_name in models:
        print(f"\n-- {model_name} --")
        sub = summary[summary["model"] == model_name]
        agg = sub.groupby("experiment")[
            ["accuracy_mean", "auroc_mean", "far_unauthorized_mean", "far_none_mean",
             "recall_anjali_mean", "recall_barath_mean"]
        ].mean().reindex(experiment_names)
        print(agg.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS.keys()), choices=list(EXPERIMENTS.keys()))
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    main(experiment_names=args.experiments, epochs=args.epochs, seeds=args.seeds, models=args.models)
