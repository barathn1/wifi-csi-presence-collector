"""Same E1_raw vs E9_paper_hampel_butterworth comparison as `run_ablation_ch6_preprocessing.py`, but
restricted to 2026-09-15 + 2026-09-16 only (2026-09-17 excluded entirely) -- direct follow-up to the
finding that Sept 17 is the outlier day (largest amplitude drift in `cross_day_eda_ch6.py`, worst
held-out fold in every preprocessing variant tried so far). With only two dates, "leave-one-out" IS
the pairwise cross-day test: train=15/test=16 and train=16/test=15.

Reuses every function from `run_ablation_ch6_preprocessing.py` unchanged (model builder, training
loop, metrics) -- only the manifest's date filter and the held-out-date loop are narrowed here.

    python3 -m ml.training.run_ablation_ch6_exclude_day17
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
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.training.run_ablation_ch6_preprocessing import CLASSES, build_model, compute_fold_metrics, \
    train_and_predict

DATES = ["2026-09-15", "2026-09-16"]
LOG_PATH = REPO_ROOT / "ml/evaluation/results/ablation_ch6_exclude_day17_log.csv"
NON_AUTH_IDX = CLASSES.index("non_auth")
ANJALI_IDX = CLASSES.index("anjali")
BARATH_IDX = CLASSES.index("barath")
LOG_FIELDS = [
    "timestamp", "experiment", "model", "held_out_date", "accuracy_mean", "accuracy_std",
    "auroc_mean", "auroc_std", "far_unauthorized_mean", "far_none_mean",
    "recall_anjali_mean", "recall_barath_mean", "n_train", "n_test", "epochs", "seeds", "notes",
    "eer_mean", "far_unauthorized_at_eer_mean", "far_none_at_eer_mean", "far_overall_at_90recall_mean",
]


def compute_threshold_metrics(probs: np.ndarray, true: np.ndarray, raw_label: np.ndarray) -> dict:
    """argmax (used by `compute_fold_metrics`) implicitly decides "authorized" whenever
    P(anjali)+P(barath) > P(non_auth) -- not a threshold tuned for anything. This instead sweeps
    every possible threshold on that same auth_score to find (a) the EER operating point (false-accept
    rate = false-reject rate, the field-standard way to report a verification system, per
    RESEARCH_NOTES.md section 7) and its FAR broken out by unauthorized vs none, and (b) the lowest
    FAR achievable while still accepting >=90% of genuine authorized windows -- a more realistic
    "how good could this be with a properly-chosen threshold" number than raw argmax gives."""
    auth_score = probs[:, ANJALI_IDX] + probs[:, BARATH_IDX]
    y_is_auth = (true != NON_AUTH_IDX).astype(int)

    best_gap = np.inf
    eer = eer_threshold = eer_far = eer_frr = float("nan")
    far_at_90recall = float("nan")
    for t in np.unique(auth_score):
        pred_is_auth = auth_score >= t
        recall = float(pred_is_auth[y_is_auth == 1].mean()) if (y_is_auth == 1).any() else float("nan")
        far = float(pred_is_auth[y_is_auth == 0].mean()) if (y_is_auth == 0).any() else float("nan")
        frr = 1.0 - recall
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, eer, eer_threshold, eer_far, eer_frr = gap, (far + frr) / 2, float(t), far, frr
        if recall >= 0.9 and (np.isnan(far_at_90recall) or far < far_at_90recall):
            far_at_90recall = far

    pred_is_auth_eer = auth_score >= eer_threshold
    mask_unauth = raw_label == "unauthorized"
    mask_none = raw_label == "none"
    far_unauth_eer = float(pred_is_auth_eer[mask_unauth].mean()) if mask_unauth.any() else float("nan")
    far_none_eer = float(pred_is_auth_eer[mask_none].mean()) if mask_none.any() else float("nan")
    return {
        "eer": eer, "far_unauthorized_at_eer": far_unauth_eer, "far_none_at_eer": far_none_eer,
        "far_overall_at_90recall": far_at_90recall,
    }


def run_experiment(config: PreprocessConfig, manifest: pd.DataFrame, epochs: int, seeds: list[int],
                    models: list[str]) -> list[dict]:
    print(f"\n=== {config.name} (2026-09-15 <-> 2026-09-16 only) ===")
    target_rate_hz = compute_dataset_target_rate_hz(manifest, config)
    window_index = build_or_load_window_index(manifest, config, target_rate_hz)

    rows = []
    for held_out in DATES:
        train_dates = [d for d in DATES if d != held_out]
        train_index = window_index[window_index["date"].isin(train_dates)].reset_index(drop=True)
        test_index = window_index[window_index["date"] == held_out].reset_index(drop=True)

        normalizer = None
        if config.normalize:
            mean, std = compute_train_normalization_stats(train_index)
            normalizer = make_normalizer(mean, std)

        train_ds = CsiWindowDataset(train_index, "taskF_identity_or_nonauth", calibration=normalizer, classes=CLASSES)
        test_ds = CsiWindowDataset(test_index, "taskF_identity_or_nonauth", calibration=normalizer, classes=CLASSES)
        raw_label = test_ds.index["label"].values

        for model_name in models:
            per_seed = []
            per_seed_thresh = []
            for seed in seeds:
                model = build_model(model_name)
                probs, true = train_and_predict(model, train_ds, test_ds, epochs, seed)
                per_seed.append(compute_fold_metrics(probs, true, raw_label))
                per_seed_thresh.append(compute_threshold_metrics(probs, true, raw_label))

            agg = {}
            for key in per_seed[0]:
                vals = np.array([m[key] for m in per_seed], dtype=np.float64)
                vals = vals[~np.isnan(vals)]
                agg[f"{key}_mean"] = float(vals.mean()) if len(vals) else float("nan")
                agg[f"{key}_std"] = float(vals.std()) if len(vals) else float("nan")
            for key in per_seed_thresh[0]:
                vals = np.array([m[key] for m in per_seed_thresh], dtype=np.float64)
                vals = vals[~np.isnan(vals)]
                agg[f"{key}_mean"] = float(vals.mean()) if len(vals) else float("nan")

            print(f"  {model_name:<12} held_out={held_out} train={train_dates} "
                  f"acc={agg['accuracy_mean']*100:.1f}% auroc={agg['auroc_mean']:.3f} "
                  f"far_unauth(argmax)={agg['far_unauthorized_mean']*100:.1f}% far_none(argmax)={agg['far_none_mean']*100:.1f}% "
                  f"recall_anjali={agg['recall_anjali_mean']*100:.1f}% recall_barath={agg['recall_barath_mean']*100:.1f}% "
                  f"|| EER={agg['eer_mean']*100:.1f}% far_unauth@EER={agg['far_unauthorized_at_eer_mean']*100:.1f}% "
                  f"far_none@EER={agg['far_none_at_eer_mean']*100:.1f}% far@90%recall={agg['far_overall_at_90recall_mean']*100:.1f}%")

            rows.append({
                "timestamp": datetime.now(timezone.utc).isoformat(), "experiment": config.name, "model": model_name,
                "held_out_date": held_out, "accuracy_mean": agg["accuracy_mean"], "accuracy_std": agg["accuracy_std"],
                "auroc_mean": agg["auroc_mean"], "auroc_std": agg["auroc_std"],
                "far_unauthorized_mean": agg["far_unauthorized_mean"], "far_none_mean": agg["far_none_mean"],
                "recall_anjali_mean": agg["recall_anjali_mean"], "recall_barath_mean": agg["recall_barath_mean"],
                "n_train": len(train_ds), "n_test": len(test_ds), "epochs": epochs, "seeds": str(seeds),
                "notes": f"train={train_dates} (2026-09-17 excluded)",
                "eer_mean": agg["eer_mean"], "far_unauthorized_at_eer_mean": agg["far_unauthorized_at_eer_mean"],
                "far_none_at_eer_mean": agg["far_none_at_eer_mean"],
                "far_overall_at_90recall_mean": agg["far_overall_at_90recall_mean"],
            })
    return rows


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=LOG_FIELDS)
    df.to_csv(LOG_PATH, mode="a" if LOG_PATH.exists() else "w", header=not LOG_PATH.exists(), index=False)


def main(experiment_names: list[str], epochs: int, seeds: list[int], models: list[str]) -> None:
    manifest = build_ch6_manifest_all_labels()
    manifest = manifest[manifest["date"].isin(DATES)].reset_index(drop=True)
    print(f"channel-6 sessions, {DATES} only: {len(manifest)}")
    print(manifest.groupby(["date", "label"]).size())

    all_rows = []
    for name in experiment_names:
        all_rows += run_experiment(EXPERIMENTS[name], manifest, epochs, seeds, models=models)
    log_rows(all_rows)
    print(f"\nlogged {len(all_rows)} rows to {LOG_PATH}")

    print("\n=== 2026-09-15 <-> 2026-09-16 SUMMARY (mean over both held-out directions) ===")
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
    p.add_argument("--experiments", nargs="+", default=["E1_raw", "E9_paper_hampel_butterworth"],
                   choices=list(EXPERIMENTS.keys()))
    p.add_argument("--models", nargs="+", default=["bilstm"], choices=["bilstm", "transformer", "mamba"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    main(experiment_names=args.experiments, epochs=args.epochs, seeds=args.seeds, models=args.models)
