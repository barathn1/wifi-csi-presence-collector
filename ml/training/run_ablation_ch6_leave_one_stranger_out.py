"""Leave-one-unauthorized-person-out on the clean 2026-09-15/16 (channel-6, day-17 excluded) subset.

Direct follow-up to the EER/threshold-tuning result (`run_ablation_ch6_exclude_day17.py`), which found
FAR-unauthorized barely moves even at the best possible decision threshold (~64-71% either way) while
FAR-none drops a lot -- suggesting the model's authorized/unauthorized confusion is a representation
problem, not a decision-rule problem. This isolates one specific hypothesis: is that confusion worse
for a stranger the model has NEVER seen in training at all (a genuine open-set case), or is it just as
bad even for pooled-training strangers? If FAR is similarly bad either way, more stranger diversity
alone probably won't fix it; if held-out-stranger FAR is much worse than the pooled numbers, data
diversity is a real lever.

Reuses the existing, general-purpose `ml.data_pipeline.splits.leave_one_unauthorized_person_out`
(already used elsewhere in this project for taskC/taskD) on top of the day15/16-only window index
already cached by `run_ablation_ch6_exclude_day17.py`'s runs -- no new preprocessing computation
needed, only new training.

    python3 -m ml.training.run_ablation_ch6_leave_one_stranger_out
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ml.data_pipeline.ablation_ch6_pipeline import build_ch6_manifest_all_labels, build_or_load_window_index, \
    compute_dataset_target_rate_hz, compute_train_normalization_stats, make_normalizer
from ml.data_pipeline.ablation_preprocessing import EXPERIMENTS, PreprocessConfig
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import leave_one_unauthorized_person_out
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.training.run_ablation_ch6_exclude_day17 import DATES, compute_threshold_metrics
from ml.training.run_ablation_ch6_preprocessing import CLASSES, build_model, compute_fold_metrics, train_and_predict

LOG_PATH = REPO_ROOT / "ml/evaluation/results/ablation_ch6_leave_one_stranger_out_log.csv"
LOG_FIELDS = [
    "timestamp", "experiment", "model", "held_out_person", "accuracy_mean", "accuracy_std",
    "auroc_mean", "auroc_std", "far_unauthorized_mean", "far_none_mean",
    "recall_anjali_mean", "recall_barath_mean", "eer_mean", "far_unauthorized_at_eer_mean",
    "far_none_at_eer_mean", "far_overall_at_90recall_mean", "n_train", "n_test", "epochs", "seeds",
]


def run_experiment(config: PreprocessConfig, manifest: pd.DataFrame, epochs: int, seeds: list[int],
                    model_name: str) -> list[dict]:
    print(f"\n=== {config.name}: leave-one-unauthorized-person-out (2026-09-15+16 pooled) ===")
    target_rate_hz = compute_dataset_target_rate_hz(manifest, config)
    window_index = build_or_load_window_index(manifest, config, target_rate_hz)

    rows = []
    for held_out_person, train_idx, test_idx in leave_one_unauthorized_person_out(window_index, include_none=True):
        train_df = window_index.iloc[train_idx].reset_index(drop=True)
        test_df = window_index.iloc[test_idx].reset_index(drop=True)

        normalizer = None
        if config.normalize:
            mean, std = compute_train_normalization_stats(train_df)
            normalizer = make_normalizer(mean, std)

        train_ds = CsiWindowDataset(train_df, "taskF_identity_or_nonauth", calibration=normalizer, classes=CLASSES)
        test_ds = CsiWindowDataset(test_df, "taskF_identity_or_nonauth", calibration=normalizer, classes=CLASSES)
        raw_label = test_ds.index["label"].values

        per_seed, per_seed_thresh = [], []
        for seed in seeds:
            model = build_model(model_name)
            probs, true = train_and_predict(model, train_ds, test_ds, epochs, seed)
            per_seed.append(compute_fold_metrics(probs, true, raw_label))
            per_seed_thresh.append(compute_threshold_metrics(probs, true, raw_label))

        agg = {}
        for src in (per_seed, per_seed_thresh):
            for key in src[0]:
                vals = np.array([m[key] for m in src], dtype=np.float64)
                vals = vals[~np.isnan(vals)]
                agg[f"{key}_mean"] = float(vals.mean()) if len(vals) else float("nan")
                agg[f"{key}_std"] = float(vals.std()) if len(vals) else float("nan")

        print(f"  held_out_person={held_out_person:<12} n_test={len(test_ds)} "
              f"acc={agg['accuracy_mean']*100:.1f}% auroc={agg['auroc_mean']:.3f} "
              f"far_unauth(argmax)={agg['far_unauthorized_mean']*100:.1f}% "
              f"far_unauth@EER={agg['far_unauthorized_at_eer_mean']*100:.1f}% "
              f"far_none@EER={agg['far_none_at_eer_mean']*100:.1f}%")

        rows.append({
            "timestamp": datetime.now(timezone.utc).isoformat(), "experiment": config.name, "model": model_name,
            "held_out_person": held_out_person, "accuracy_mean": agg["accuracy_mean"],
            "accuracy_std": agg["accuracy_std"], "auroc_mean": agg["auroc_mean"], "auroc_std": agg["auroc_std"],
            "far_unauthorized_mean": agg["far_unauthorized_mean"], "far_none_mean": agg["far_none_mean"],
            "recall_anjali_mean": agg["recall_anjali_mean"], "recall_barath_mean": agg["recall_barath_mean"],
            "eer_mean": agg["eer_mean"], "far_unauthorized_at_eer_mean": agg["far_unauthorized_at_eer_mean"],
            "far_none_at_eer_mean": agg["far_none_at_eer_mean"],
            "far_overall_at_90recall_mean": agg["far_overall_at_90recall_mean"],
            "n_train": len(train_ds), "n_test": len(test_ds), "epochs": epochs, "seeds": str(seeds),
        })
    return rows


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=LOG_FIELDS)
    df.to_csv(LOG_PATH, mode="a" if LOG_PATH.exists() else "w", header=not LOG_PATH.exists(), index=False)


def main(experiment_names: list[str], epochs: int, seeds: list[int], model_name: str) -> None:
    manifest = build_ch6_manifest_all_labels()
    manifest = manifest[manifest["date"].isin(DATES)].reset_index(drop=True)
    print(f"channel-6 sessions, {DATES} only: {len(manifest)}")
    print(manifest.groupby(["date", "label"]).size())

    all_rows = []
    for name in experiment_names:
        all_rows += run_experiment(EXPERIMENTS[name], manifest, epochs, seeds, model_name)
    log_rows(all_rows)
    print(f"\nlogged {len(all_rows)} rows to {LOG_PATH}")

    print("\n=== SUMMARY: mean FAR-unauthorized@EER over all held-out strangers, per experiment ===")
    summary = pd.DataFrame(all_rows)
    print(summary.groupby("experiment")[
        ["far_unauthorized_at_eer_mean", "far_none_at_eer_mean", "auroc_mean"]
    ].agg(["mean", "min", "max"]).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=["E1_raw", "E9_paper_hampel_butterworth"],
                   choices=list(EXPERIMENTS.keys()))
    p.add_argument("--model", default="bilstm", choices=["bilstm", "transformer", "mamba"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = p.parse_args()
    main(experiment_names=args.experiments, epochs=args.epochs, seeds=args.seeds, model_name=args.model)
