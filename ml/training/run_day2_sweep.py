"""Day 2 comprehensive sweep -- see ml/reports/day2_findings.md for the write-up. Five stages:

  Stage B  -- try every model in the zoo (RandomForest, SVM, CNN1D, ResNet1D, BiLSTM, WhoFi, DualBranch,
              CrossAttn, CalibContext) at default config on taskD, both split types, raw+calibA. The
              "does it work at all" survey.
  Stage T1 -- architecture ablation on the attention models only: depth (num_layers), heads, feed-
              forward width, pre/post-LN, one-at-a-time off a fixed baseline, grounded in "Attention Is
              All You Need". Day-disjoint, raw only (isolates architecture from calibration).
  Stage T2 -- full-rigor validation of the single best config found so far (Stage B + T1 combined):
              session-disjoint 5-fold AND both day-disjoint directions, raw/calibA/calibB (calibB only
              meaningful day-disjoint), with false-accept-by-negative-class breakdown. The headline
              auth-vs-non-auth number.
  Stage T3 -- secondary tasks (presence, standing/walking, threeway, identity, open-set proxy) with the
              best config only, day-disjoint, raw+calibA.
  Stage S  -- does deciding from a longer stretch (10-30s of consecutive windows) beat one ~1s window?
              Both plain score-averaging (already in evaluation/segment_aggregation.py) AND a learned
              attention-pooling second stage (models/segment_pooling.py) across several segment lengths.
  Stage N  -- is resampling every session onto a shared 128-subcarrier grid actually hurting accuracy
              vs. each day's own native resolution? Single-day-only (native mode can't cross days), so
              this is a bounded side check, not plumbed through the whole sweep.

Every stage appends rows to `ml/evaluation/results/day2_sweep_log.csv` as it goes (not just at the end)
so a partial run is never wasted.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.data_pipeline.calibration import apply_variant_a, apply_variant_b, compute_day_baseline
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.features import build_feature_matrix
from ml.data_pipeline.splits import assert_no_group_leakage, day_disjoint_split, session_disjoint_kfold
from ml.data_pipeline.tasks import TASKS
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.data_pipeline.windowing import load_manifest
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.evaluation.segment_aggregation import aggregate_scores_by_session
from ml.models.baselines import make_baseline
from ml.models.cnn1d import CNN1DDualBranch
from ml.models.lstm import BiLSTMDualBranch
from ml.models.resnet1d import ResNet1DDualBranch
from ml.models.segment_pooling import AttentionPoolClassifier
from ml.models.svm import make_svm
from ml.models.transformer_calib_context import CalibrationContextTransformer
from ml.models.transformer_crossattn import CrossAttentionTransformer
from ml.models.transformer_dualbranch import DualBranchTransformer
from ml.models.transformer_whofi import WhoFiTransformer
from ml.training.train import DEVICE, train_classifier

CACHE_DIR = REPO_ROOT / "ml/data_pipeline/cache"
LOG_PATH = REPO_ROOT / "ml/evaluation/results/day2_sweep_log.csv"
INDEX_PATH = CACHE_DIR / "window_index_w200_s100.csv"          # resampled, 128 subcarriers, both days
NATIVE_INDEX_PATH = CACHE_DIR / "window_index_w200_s100_native.csv"
N_SUB_RESAMPLED = 128

FIELDNAMES = [
    "timestamp", "stage", "task", "preprocessing", "model", "split_type", "train_date", "test_date",
    "fold", "seed", "num_layers", "n_heads", "d_ff", "norm_first", "accuracy", "eer", "auroc",
    "false_accept_unauthorized", "false_accept_none", "n_train", "n_test", "notes",
]

ATTENTION_MODELS = ["whofi_transformer", "dualbranch_transformer", "crossattn_transformer", "calib_context_transformer"]
ALL_NEURAL_MODELS = ["cnn1d", "resnet1d", "bilstm"] + ATTENTION_MODELS
ALL_CLASSICAL_MODELS = ["random_forest", "svm"]


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})


# ---------------------------------------------------------------- calibration wiring ----

def calibA_fn(manifest: pd.DataFrame, dates: list[str], mode: str = "resampled"):
    baselines = {d: compute_day_baseline(manifest, d, mode=mode) for d in dates}

    def fn(amp, phase, row):
        return apply_variant_a(amp, phase, baselines[row["date"]])
    return fn


def calibB_fn(manifest: pd.DataFrame, train_date: str, test_date: str, mode: str = "resampled"):
    baselines = {d: compute_day_baseline(manifest, d, mode=mode) for d in {train_date, test_date}}
    ref = baselines[train_date]

    def fn(amp, phase, row):
        if row["date"] == train_date:
            return apply_variant_a(amp, phase, ref)
        return apply_variant_b(amp, phase, baseline_today=baselines[row["date"]], baseline_ref=ref)
    return fn


def get_calibration(preprocessing: str, manifest, dates, mode="resampled", train_date=None, test_date=None):
    if preprocessing == "raw":
        return None
    if preprocessing == "calibA":
        return calibA_fn(manifest, dates, mode)
    if preprocessing == "calibB":
        assert train_date and test_date, "calibB needs a direction (train_date/test_date)"
        return calibB_fn(manifest, train_date, test_date, mode)
    raise ValueError(preprocessing)


# ---------------------------------------------------------------- model construction ----

def build_neural_model(model_name: str, n_sub: int, n_classes: int, arch_kwargs: dict | None = None):
    kw = arch_kwargs or {}
    if model_name == "cnn1d":
        return CNN1DDualBranch(n_sub, n_classes)
    if model_name == "resnet1d":
        return ResNet1DDualBranch(n_sub, n_classes)
    if model_name == "bilstm":
        return BiLSTMDualBranch(n_sub, n_classes, **{k: v for k, v in kw.items() if k in ("hidden_size", "num_layers")})
    if model_name == "whofi_transformer":
        return WhoFiTransformer(n_sub, n_classes, **kw)
    if model_name == "dualbranch_transformer":
        return DualBranchTransformer(n_sub, n_classes, **kw)
    if model_name == "crossattn_transformer":
        return CrossAttentionTransformer(n_sub, n_classes, **kw)
    if model_name == "calib_context_transformer":
        return CalibrationContextTransformer(n_sub, n_classes, **kw)
    raise ValueError(model_name)


CLASSICAL_FACTORIES = {"random_forest": make_baseline, "svm": make_svm}


# ---------------------------------------------------------------- evaluation helpers ----

@torch.no_grad()
def neural_logits(model: torch.nn.Module, test_ds: CsiWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    all_logits, all_y = [], []
    for amp, phase, y in loader:
        logits = model(amp.to(DEVICE), phase.to(DEVICE))
        all_logits.append(logits.numpy())
        all_y.append(np.asarray(y))
    return np.concatenate(all_logits), np.concatenate(all_y)


def binary_metrics(logits: np.ndarray, y_true: np.ndarray, original_labels: np.ndarray | None = None) -> dict:
    proba = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    pred = logits.argmax(axis=-1)
    eer, _ = compute_eer(y_true, proba)
    auroc = compute_auroc(y_true, proba)
    out = {"accuracy": float((pred == y_true).mean()), "eer": eer, "auroc": auroc}
    if original_labels is not None:
        for neg in ("unauthorized", "none"):
            m = original_labels == neg
            if m.sum() > 0:
                out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())
    return out


def multiclass_metrics(logits: np.ndarray, y_true: np.ndarray) -> dict:
    pred = logits.argmax(axis=-1)
    return {"accuracy": float((pred == y_true).mean())}


NEURAL_MAX_TRAIN = 4000  # attention models cost ~O(T^2) per window regardless of how many windows exist;
# capping keeps wall-clock per run bounded without touching model quality much given how redundant
# 50%-overlapping windows already are. (Measured ~0.6-1.8s/batch for the attention models on this CPU --
# at the full instance/seed/epoch count this script sweeps, 8000 made a full run ~20-30h; 4000 halves it.)


def run_neural_instance(model_name, task_name, window_index, train_idx, test_idx, calibration, n_sub,
                         seed, epochs, arch_kwargs=None, calib_context_baseline=None):
    full_ds = CsiWindowDataset(window_index, task_name, calibration=calibration)
    if len(train_idx) > NEURAL_MAX_TRAIN:
        train_idx = np.random.default_rng(seed).choice(train_idx, size=NEURAL_MAX_TRAIN, replace=False)
    train_ds = full_ds.subset_by_index_rows(train_idx)
    test_ds = full_ds.subset_by_index_rows(test_idx)
    n_classes = len(full_ds.classes)

    model = build_neural_model(model_name, n_sub, n_classes, arch_kwargs)
    if model_name == "calib_context_transformer":
        b = calib_context_baseline
        model.set_calibration_context(b.amp_mean, b.amp_std, b.phase_mean, b.phase_std)

    result = train_classifier(model, train_ds, test_ds, epochs=epochs, seed=seed)
    logits, y_true = neural_logits(model, test_ds)
    if n_classes == 2:
        original_labels = test_ds.index["label"].values if task_name == "taskD_auth_vs_nonauth" else None
        metrics = binary_metrics(logits, y_true, original_labels)
    else:
        metrics = multiclass_metrics(logits, y_true)
    metrics["n_train"], metrics["n_test"] = result["n_train"], result["n_test"]
    return metrics, model, test_ds


SVM_MAX_TRAIN = 3000  # RBF-SVM fit time grows ~cubically with n; unbounded n (~24k/fold here) never finishes


def run_classical_instance(model_name, task_name, window_index, X_full, train_idx, test_idx, seed):
    mask, y = TASKS[task_name](window_index)
    sub_index = window_index[mask].reset_index(drop=True)
    X = X_full[mask]
    y = np.asarray(y)
    classes = sorted(set(y.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[v] for v in y])

    fit_idx = train_idx
    if model_name == "svm" and len(train_idx) > SVM_MAX_TRAIN:
        fit_idx = np.random.default_rng(seed).choice(train_idx, size=SVM_MAX_TRAIN, replace=False)

    model = CLASSICAL_FACTORIES[model_name](seed=seed)
    model.fit(X[fit_idx], y_idx[fit_idx])
    pred = model.predict(X[test_idx])
    n_classes = len(classes)

    if n_classes == 2:
        # SVC is fit with probability=False (Platt-scaling calibration is too slow at this row count) --
        # decision_function's signed margin distance is monotonic with confidence, fine for ROC metrics.
        proba = model.predict_proba(X[test_idx])[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X[test_idx])
        eer, _ = compute_eer(y_idx[test_idx], proba)
        auroc = compute_auroc(y_idx[test_idx], proba)
        metrics = {"accuracy": float((pred == y_idx[test_idx]).mean()), "eer": eer, "auroc": auroc}
        if task_name == "taskD_auth_vs_nonauth":
            original_labels = sub_index.iloc[test_idx]["label"].values
            for neg in ("unauthorized", "none"):
                m = original_labels == neg
                if m.sum() > 0:
                    metrics[f"false_accept_{neg}"] = float((pred[m] == 1).mean())
    else:
        metrics = {"accuracy": float((pred == y_idx[test_idx]).mean())}
    metrics["n_train"], metrics["n_test"] = len(fit_idx), len(test_idx)
    return metrics


def row_template(**kw) -> dict:
    row = {"timestamp": datetime.now(timezone.utc).isoformat()}
    row.update(kw)
    return row


# ------------------------------------------------------------------------ Stage B ----

def stage_b(window_index, manifest, dates, epochs, seeds_neural=(0, 1), smoke_test=False) -> None:
    task_name = "taskD_auth_vs_nonauth"
    rows = []
    classical_models = ALL_CLASSICAL_MODELS[:1] if smoke_test else ALL_CLASSICAL_MODELS
    # smoke test still needs to touch calib_context_transformer's distinct code path (context buffer),
    # not just the first N in list order.
    neural_models = ["cnn1d", "crossattn_transformer", "calib_context_transformer"] if smoke_test else ALL_NEURAL_MODELS
    max_folds = 1 if smoke_test else 5

    # --- session-disjoint 5-fold (pools both days) ---
    calib_cache = {"raw": None, "calibA": calibA_fn(manifest, dates)}
    full_ds_probe = CsiWindowDataset(window_index, task_name)
    for preprocessing in ("raw", "calibA"):
        calibration = calib_cache[preprocessing]
        X_full = classical_features_for(preprocessing, window_index, manifest, dates)
        folds = list(session_disjoint_kfold(full_ds_probe.index, n_splits=5))[:max_folds]
        for fold, (train_idx, test_idx) in enumerate(folds):
            assert_no_group_leakage(full_ds_probe.index, train_idx, test_idx, "session_dir")
            for model_name in classical_models:
                m = run_classical_instance(model_name, task_name, window_index, X_full, train_idx, test_idx, seed=0)
                rows.append(row_template(stage="B", task=task_name, preprocessing=preprocessing, model=model_name,
                                          split_type="session_disjoint", fold=fold, seed=0, **m))
            for model_name in neural_models:
                if model_name == "calib_context_transformer":
                    continue  # no single context makes sense in a mixed-date fold, see Stage T2/T3 instead
                for seed in seeds_neural:
                    m, _, _ = run_neural_instance(model_name, task_name, window_index, train_idx, test_idx,
                                                    calibration, N_SUB_RESAMPLED, seed, epochs)
                    rows.append(row_template(stage="B", task=task_name, preprocessing=preprocessing, model=model_name,
                                              split_type="session_disjoint", fold=fold, seed=seed, **m))
            log_rows(rows); rows = []
            print(f"  [StageB session_disjoint fold {fold} {preprocessing}] logged", flush=True)

    # --- day-disjoint, both directions ---
    for train_date, test_date in [(dates[0], dates[1]), (dates[1], dates[0])]:
        # day_disjoint_split always tests on the LAST sorted date; get the OTHER direction by picking
        # train/test rows directly from `date` instead.
        full_ds = CsiWindowDataset(window_index, task_name)
        train_idx = np.flatnonzero((full_ds.index["date"] == train_date).values)
        test_idx = np.flatnonzero((full_ds.index["date"] == test_date).values)
        assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")

        for preprocessing in ("raw", "calibA", "calibB"):
            calibration = get_calibration(preprocessing, manifest, dates, train_date=train_date, test_date=test_date)
            for model_name in classical_models:
                X_full = classical_features_for(preprocessing, window_index, manifest, dates, train_date, test_date)
                m = run_classical_instance(model_name, task_name, window_index, X_full, train_idx, test_idx, seed=0)
                rows.append(row_template(stage="B", task=task_name, preprocessing=preprocessing, model=model_name,
                                          split_type="day_disjoint", train_date=train_date, test_date=test_date,
                                          fold=0, seed=0, **m))
            for model_name in neural_models:
                calib_baseline = None
                if model_name == "calib_context_transformer":
                    calib_baseline = compute_day_baseline(manifest, train_date, mode="resampled")
                for seed in seeds_neural:
                    m, _, _ = run_neural_instance(model_name, task_name, window_index, train_idx, test_idx,
                                                    calibration, N_SUB_RESAMPLED, seed, epochs,
                                                    calib_context_baseline=calib_baseline)
                    rows.append(row_template(stage="B", task=task_name, preprocessing=preprocessing, model=model_name,
                                              split_type="day_disjoint", train_date=train_date, test_date=test_date,
                                              fold=0, seed=seed, **m))
            log_rows(rows); rows = []
            print(f"  [StageB day_disjoint train={train_date} test={test_date} {preprocessing}] logged", flush=True)


_CLASSICAL_FEATURE_CACHE: dict[str, np.ndarray] = {}


def classical_features_for(preprocessing, window_index, manifest, dates, train_date=None, test_date=None) -> np.ndarray:
    if preprocessing in ("raw", "calibA"):
        path = CACHE_DIR / f"features_{preprocessing}_window_index_w200_s100.npy"
        if preprocessing not in _CLASSICAL_FEATURE_CACHE:
            _CLASSICAL_FEATURE_CACHE[preprocessing] = np.load(path)
        return _CLASSICAL_FEATURE_CACHE[preprocessing]

    key = f"calibB_train{train_date}"
    if key not in _CLASSICAL_FEATURE_CACHE:
        cache_path = CACHE_DIR / f"features_{key}.npy"
        if cache_path.exists():
            _CLASSICAL_FEATURE_CACHE[key] = np.load(cache_path)
        else:
            calibration = calibB_fn(manifest, train_date, test_date, mode="resampled")
            X = build_feature_matrix(window_index, calibration=calibration)
            np.save(cache_path, X)
            _CLASSICAL_FEATURE_CACHE[key] = X
    return _CLASSICAL_FEATURE_CACHE[key]


# ----------------------------------------------------------------------- Stage T1 ----

T1_BASELINE = dict(num_layers=1, n_heads=4, d_ff=64, norm_first=False)
T1_VARIATIONS = [
    dict(num_layers=2), dict(num_layers=4),
    dict(n_heads=2), dict(n_heads=8),
    dict(d_ff=128),
    dict(norm_first=True),
]


def stage_t1(window_index, manifest, dates, epochs, seeds=(0, 1, 2), smoke_test=False) -> list[dict]:
    task_name = "taskD_auth_vs_nonauth"
    train_date, test_date = dates[0], dates[1]
    full_ds = CsiWindowDataset(window_index, task_name)
    train_idx = np.flatnonzero((full_ds.index["date"] == train_date).values)
    test_idx = np.flatnonzero((full_ds.index["date"] == test_date).values)
    assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")

    variations = T1_VARIATIONS[:1] if smoke_test else T1_VARIATIONS
    configs = [dict(T1_BASELINE)] + [dict(T1_BASELINE, **v) for v in variations]
    attention_models = ATTENTION_MODELS[:2] if smoke_test else ATTENTION_MODELS
    results_summary = []
    for model_name in attention_models:
        calib_baseline = compute_day_baseline(manifest, train_date, mode="resampled") if model_name == "calib_context_transformer" else None
        for cfg in configs:
            accs = []
            rows = []
            for seed in seeds:
                m, _, _ = run_neural_instance(model_name, task_name, window_index, train_idx, test_idx,
                                                None, N_SUB_RESAMPLED, seed, epochs, arch_kwargs=cfg,
                                                calib_context_baseline=calib_baseline)
                accs.append(m.get("auroc") if not np.isnan(m.get("auroc", np.nan)) else m["accuracy"])
                rows.append(row_template(stage="T1", task=task_name, preprocessing="raw", model=model_name,
                                          split_type="day_disjoint", train_date=train_date, test_date=test_date,
                                          fold=0, seed=seed, **cfg, **m))
            log_rows(rows)
            results_summary.append({"model": model_name, **cfg, "mean_score": float(np.nanmean(accs))})
            print(f"  [StageT1] {model_name} {cfg} mean_score={np.nanmean(accs):.4f}", flush=True)
    return results_summary


# ------------------------------------------------------------- Stage T1b (bilstm) ----

# bilstm was Stage B's single strongest, most direction-stable model (see day2_next_steps.md item 5)
# but never got an architecture ablation the way the attention models did in Stage T1 -- this gives it
# the same one-at-a-time treatment over its own relevant axes (hidden size, LSTM depth).
T1B_BASELINE = dict(hidden_size=32, num_layers=1)
T1B_VARIATIONS = [dict(hidden_size=16), dict(hidden_size=64), dict(num_layers=2), dict(num_layers=3)]


def stage_t1_bilstm(window_index, manifest, dates, epochs, seeds=(0, 1, 2), smoke_test=False) -> list[dict]:
    task_name = "taskD_auth_vs_nonauth"
    train_date, test_date = dates[0], dates[1]
    full_ds = CsiWindowDataset(window_index, task_name)
    train_idx = np.flatnonzero((full_ds.index["date"] == train_date).values)
    test_idx = np.flatnonzero((full_ds.index["date"] == test_date).values)
    assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")

    variations = T1B_VARIATIONS[:1] if smoke_test else T1B_VARIATIONS
    configs = [dict(T1B_BASELINE)] + [dict(T1B_BASELINE, **v) for v in variations]
    results_summary = []
    for cfg in configs:
        accs, rows = [], []
        for seed in seeds:
            m, _, _ = run_neural_instance("bilstm", task_name, window_index, train_idx, test_idx,
                                            None, N_SUB_RESAMPLED, seed, epochs, arch_kwargs=cfg)
            accs.append(m.get("auroc") if not np.isnan(m.get("auroc", np.nan)) else m["accuracy"])
            rows.append(row_template(stage="T1b", task=task_name, preprocessing="raw", model="bilstm",
                                      split_type="day_disjoint", train_date=train_date, test_date=test_date,
                                      fold=0, seed=seed, num_layers=cfg["num_layers"], d_ff=cfg["hidden_size"],
                                      notes=f"d_ff column repurposed as hidden_size={cfg['hidden_size']}", **m))
        log_rows(rows)
        results_summary.append({"model": "bilstm", **cfg, "mean_score": float(np.nanmean(accs))})
        print(f"  [StageT1b] bilstm {cfg} mean_score={np.nanmean(accs):.4f}", flush=True)
    return results_summary


# ----------------------------------------------------------------------- Stage T2 ----

def stage_t2(window_index, manifest, dates, best_model, best_cfg, epochs, seeds=(0, 1, 2), smoke_test=False) -> None:
    task_name = "taskD_auth_vs_nonauth"
    calib_baseline_by_date = {d: compute_day_baseline(manifest, d, mode="resampled") for d in dates}
    rows = []

    full_ds = CsiWindowDataset(window_index, task_name)
    folds = list(session_disjoint_kfold(full_ds.index, n_splits=5))[:1 if smoke_test else 5]
    for fold, (train_idx, test_idx) in enumerate(folds):
        assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
        for preprocessing in ("raw", "calibA"):
            calibration = get_calibration(preprocessing, manifest, dates)
            for seed in seeds:
                calib_baseline = calib_baseline_by_date[dates[0]] if best_model == "calib_context_transformer" else None
                m, _, _ = run_neural_instance(best_model, task_name, window_index, train_idx, test_idx,
                                                calibration, N_SUB_RESAMPLED, seed, epochs, arch_kwargs=best_cfg,
                                                calib_context_baseline=calib_baseline)
                rows.append(row_template(stage="T2", task=task_name, preprocessing=preprocessing, model=best_model,
                                          split_type="session_disjoint", fold=fold, seed=seed, **best_cfg, **m))
        log_rows(rows); rows = []
        print(f"  [StageT2 session_disjoint fold {fold}] logged", flush=True)

    for train_date, test_date in [(dates[0], dates[1]), (dates[1], dates[0])]:
        train_idx = np.flatnonzero((full_ds.index["date"] == train_date).values)
        test_idx = np.flatnonzero((full_ds.index["date"] == test_date).values)
        assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
        for preprocessing in ("raw", "calibA", "calibB"):
            calibration = get_calibration(preprocessing, manifest, dates, train_date=train_date, test_date=test_date)
            for seed in seeds:
                calib_baseline = calib_baseline_by_date[train_date] if best_model == "calib_context_transformer" else None
                m, _, _ = run_neural_instance(best_model, task_name, window_index, train_idx, test_idx,
                                                calibration, N_SUB_RESAMPLED, seed, epochs, arch_kwargs=best_cfg,
                                                calib_context_baseline=calib_baseline)
                rows.append(row_template(stage="T2", task=task_name, preprocessing=preprocessing, model=best_model,
                                          split_type="day_disjoint", train_date=train_date, test_date=test_date,
                                          fold=0, seed=seed, **best_cfg, **m))
        log_rows(rows); rows = []
        print(f"  [StageT2 day_disjoint train={train_date} test={test_date}] logged", flush=True)


# ----------------------------------------------------------------------- Stage T3 ----

SECONDARY_TASKS = ["task0_presence", "taskE_motion_standing_vs_walking", "taskA_threeway",
                   "taskB_identity", "taskC_openset_proxy"]


def stage_t3(window_index, manifest, dates, best_model, best_cfg, epochs, seed=0) -> None:
    train_date, test_date = dates[0], dates[1]
    rows = []
    for task_name in SECONDARY_TASKS:
        full_ds = CsiWindowDataset(window_index, task_name)
        train_idx = np.flatnonzero((full_ds.index["date"] == train_date).values)
        test_idx = np.flatnonzero((full_ds.index["date"] == test_date).values)
        if len(train_idx) == 0 or len(test_idx) == 0 or len(set(full_ds.y[test_idx].tolist())) < 2:
            print(f"  [StageT3] skipping {task_name}: degenerate day-disjoint split for this task")
            continue
        assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
        for preprocessing in ("raw", "calibA"):
            calibration = get_calibration(preprocessing, manifest, dates, train_date=train_date, test_date=test_date)
            m, _, _ = run_neural_instance(best_model, task_name, window_index, train_idx, test_idx,
                                            calibration, N_SUB_RESAMPLED, seed, epochs, arch_kwargs=best_cfg)
            rows.append(row_template(stage="T3", task=task_name, preprocessing=preprocessing, model=best_model,
                                      split_type="day_disjoint", train_date=train_date, test_date=test_date,
                                      fold=0, seed=seed, **best_cfg, **m))
        print(f"  [StageT3] {task_name} done", flush=True)
    log_rows(rows)


# ------------------------------------------------------------------------ Stage S ----

SEGMENT_LENGTHS = [(10, "~5s"), (20, "~10s"), (40, "~20s"), (60, "~30s")]


def stage_s(window_index, manifest, dates, best_model, best_cfg, epochs, seed=0) -> None:
    task_name = "taskD_auth_vs_nonauth"
    train_date, test_date = dates[0], dates[1]
    full_ds = CsiWindowDataset(window_index, task_name)
    train_idx = np.flatnonzero((full_ds.index["date"] == train_date).values)
    test_idx = np.flatnonzero((full_ds.index["date"] == test_date).values)
    assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")

    # raw AND calibA: calibA has been the strongest single lever everywhere else in this sweep (see
    # day2_next_steps.md) -- Stage S originally only ever tested raw, which was a real gap, not a
    # deliberate choice (fixed 2026-09-12).
    for preprocessing in ("raw", "calibA"):
        calibration = get_calibration(preprocessing, manifest, dates, train_date=train_date, test_date=test_date)
        rows = []
        m, model, test_ds = run_neural_instance(best_model, task_name, window_index, train_idx, test_idx,
                                                  calibration, N_SUB_RESAMPLED, seed, epochs, arch_kwargs=best_cfg)
        logits, y_true = neural_logits(model, test_ds)
        proba = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
        rows.append(row_template(stage="S", task=task_name, preprocessing=preprocessing, model=best_model,
                                  split_type="day_disjoint_no_aggregation", train_date=train_date, test_date=test_date,
                                  fold=0, seed=seed, notes="n_windows=1 (baseline)", **m, **best_cfg))

        for n_windows, label in SEGMENT_LENGTHS:
            segments = aggregate_scores_by_session(test_ds.index, proba, n_windows=n_windows)
            if len(segments) == 0:
                continue
            y_seg = (segments["label"] == "authorized").astype(int).values
            pred_seg = (segments["aggregated_score"].values >= 0.5).astype(int)
            eer, _ = compute_eer(y_seg, segments["aggregated_score"].values)
            auroc = compute_auroc(y_seg, segments["aggregated_score"].values)
            seg_metrics = {"accuracy": float((pred_seg == y_seg).mean()), "eer": eer, "auroc": auroc,
                           "n_train": m["n_train"], "n_test": len(segments)}
            for neg in ("unauthorized", "none"):
                nm = segments["label"].values == neg
                if nm.sum() > 0:
                    seg_metrics[f"false_accept_{neg}"] = float(pred_seg[nm].mean())
            rows.append(row_template(stage="S", task=task_name, preprocessing=preprocessing, model=best_model,
                                      split_type="day_disjoint_mean_aggregation", train_date=train_date,
                                      test_date=test_date, fold=0, seed=seed, notes=f"n_windows={n_windows} ({label})",
                                      **seg_metrics, **best_cfg))

        # learned attention-pooling second stage: reuse the trained base model's embed() on train+test
        # windows, group into the same segment lengths, then train AttentionPoolClassifier on TRAIN
        # segments' embeddings.
        if hasattr(model, "embed"):
            train_embeds, train_meta = _embed_all(model, full_ds.subset_by_index_rows(train_idx))
            test_embeds, test_meta = _embed_all(model, test_ds)
            for n_windows, label in SEGMENT_LENGTHS:
                train_seq, train_y = _group_embeddings(train_embeds, train_meta, n_windows)
                test_seq, test_y, test_labels = _group_embeddings(test_embeds, test_meta, n_windows, return_labels=True)
                if len(train_seq) < 8 or len(test_seq) == 0 or len(set(test_y.tolist())) < 2:
                    continue
                pool_metrics = _train_attention_pool(train_seq, train_y, test_seq, test_y, test_labels, seed=seed)
                rows.append(row_template(stage="S", task=task_name, preprocessing=preprocessing,
                                          model=f"{best_model}+attn_pool", split_type="day_disjoint_attention_pool",
                                          train_date=train_date, test_date=test_date, fold=0, seed=seed,
                                          notes=f"n_windows={n_windows} ({label})", **pool_metrics, **best_cfg))
        log_rows(rows)
        print(f"  [StageS] {preprocessing} done", flush=True)


@torch.no_grad()
def _embed_all(model, ds: CsiWindowDataset):
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    embeds = []
    for amp, phase, _ in loader:
        embeds.append(model.embed(amp.to(DEVICE), phase.to(DEVICE)).numpy())
    return np.concatenate(embeds), ds.index.reset_index(drop=True)


def _group_embeddings(embeds: np.ndarray, meta: pd.DataFrame, n_windows: int, return_labels: bool = False):
    seqs, ys, labels = [], [], []
    for session_dir, group in meta.groupby("session_dir", sort=False):
        group = group.sort_values("start")
        idx = group.index.values
        for i in range(0, len(idx) - n_windows + 1, n_windows):
            sel = idx[i:i + n_windows]
            seqs.append(embeds[sel])
            ys.append(int(group.loc[sel[0], "label"] == "authorized"))
            labels.append(group.loc[sel[0], "label"])
    if not seqs:
        return np.empty((0, n_windows, embeds.shape[1]), dtype=np.float32), np.empty((0,), dtype=int), []
    out = np.stack(seqs).astype(np.float32), np.array(ys)
    return (*out, np.array(labels)) if return_labels else out


def _train_attention_pool(train_seq, train_y, test_seq, test_y, test_labels, seed, epochs=10):
    torch.manual_seed(seed)
    embed_dim = train_seq.shape[-1]
    n_heads = 4 if embed_dim % 4 == 0 else 1
    pool = AttentionPoolClassifier(embed_dim, 2, n_heads=n_heads)
    opt = torch.optim.Adam(pool.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    X_train = torch.from_numpy(train_seq)
    y_train = torch.from_numpy(train_y).long()
    for _ in range(epochs):
        pool.train()
        opt.zero_grad()
        logits = pool(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

    pool.eval()
    with torch.no_grad():
        logits = pool(torch.from_numpy(test_seq)).numpy()
    proba = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    pred = logits.argmax(axis=-1)
    eer, _ = compute_eer(test_y, proba)
    auroc = compute_auroc(test_y, proba)
    metrics = {"accuracy": float((pred == test_y).mean()), "eer": eer, "auroc": auroc,
               "n_train": len(train_seq), "n_test": len(test_seq)}
    for neg in ("unauthorized", "none"):
        m = test_labels == neg
        if m.sum() > 0:
            metrics[f"false_accept_{neg}"] = float((pred[m] == 1).mean())
    return metrics


# ------------------------------------------------------------------------ Stage N ----

def stage_n(epochs, best_model, best_cfg, seeds=(0, 1)) -> None:
    """Native (per-day own subcarrier count) vs resampled (128-shared), same-day-only session-disjoint
    5-fold, one day at a time -- resampling can't be compared cross-day since native mode makes the two
    days shape-incompatible by construction."""
    manifest = load_manifest()
    native_index = pd.read_csv(NATIVE_INDEX_PATH)
    resampled_index = pd.read_csv(INDEX_PATH)
    task_name = "taskD_auth_vs_nonauth"

    for date in sorted(native_index["date"].unique()):
        native_day = native_index[native_index["date"] == date].reset_index(drop=True)
        resampled_day = resampled_index[resampled_index["date"] == date].reset_index(drop=True)
        with np.load(native_day["cache_path"].iloc[0]) as d:
            n_sub_native = d["amplitude"].shape[1]

        rows = []
        for mode_name, idx_df, n_sub in [("native", native_day, n_sub_native), ("resampled", resampled_day, N_SUB_RESAMPLED)]:
            full_ds = CsiWindowDataset(idx_df, task_name)
            if len(full_ds.classes) < 2:
                continue
            for fold, (train_idx, test_idx) in enumerate(session_disjoint_kfold(full_ds.index, n_splits=5)):
                assert_no_group_leakage(full_ds.index, train_idx, test_idx, "session_dir")
                for seed in seeds:
                    m, _, _ = run_neural_instance(best_model, task_name, idx_df, train_idx, test_idx,
                                                    None, n_sub, seed, epochs, arch_kwargs=best_cfg)
                    rows.append(row_template(stage="N", task=task_name, preprocessing=mode_name, model=best_model,
                                              split_type=f"session_disjoint_{date}_only", fold=fold, seed=seed,
                                              **best_cfg, **m))
        log_rows(rows)
        print(f"  [StageN] {date} done", flush=True)


# ------------------------------------------------------------------------------------

def main(epochs: int, smoke_test: bool) -> None:
    manifest = load_manifest()
    window_index = pd.read_csv(INDEX_PATH)
    dates = sorted(window_index["date"].unique())
    assert len(dates) == 2, dates

    if smoke_test:
        epochs = 1
        print(f"SMOKE TEST: full {len(window_index)}-window dataset, epochs={epochs}, "
              f"truncated model/config/fold lists")

    print("=== Stage B: comprehensive model survey ===", flush=True)
    # Stage B is the breadth-over-depth survey (Stage T1/T2 below add the real seed-variance rigor on
    # the winning config) -- 1 seed here, matching how the original single-day scripts (run_stage2.py
    # etc.) sized this same tradeoff.
    stage_b(window_index, manifest, dates, epochs, seeds_neural=(0,), smoke_test=smoke_test)

    print("=== Stage T1: transformer architecture ablation ===", flush=True)
    t1_results = stage_t1(window_index, manifest, dates, epochs, seeds=(0,) if smoke_test else (0, 1),
                           smoke_test=smoke_test)

    print("=== Stage T1b: bilstm architecture ablation ===", flush=True)
    t1b_results = stage_t1_bilstm(window_index, manifest, dates, epochs, seeds=(0,) if smoke_test else (0, 1),
                                   smoke_test=smoke_test)

    best = max(t1_results + t1b_results, key=lambda r: r["mean_score"])
    best_model = best["model"]
    if best_model == "bilstm":
        best_cfg = {k: best[k] for k in ("hidden_size", "num_layers")}
    else:
        best_cfg = {k: best[k] for k in ("num_layers", "n_heads", "d_ff", "norm_first")}
    print(f"Stage T1/T1b winner: {best_model} {best_cfg} (mean_score={best['mean_score']:.4f})")

    print("=== Stage T2: full-rigor validation of best config ===", flush=True)
    stage_t2(window_index, manifest, dates, best_model, best_cfg, epochs, seeds=(0,) if smoke_test else (0, 1, 2),
             smoke_test=smoke_test)

    print("=== Stage T3: secondary tasks ===", flush=True)
    stage_t3(window_index, manifest, dates, best_model, best_cfg, epochs)

    print("=== Stage S: segment-length aggregation ===", flush=True)
    stage_s(window_index, manifest, dates, best_model, best_cfg, epochs)

    if not smoke_test:
        print("=== Stage N: native vs resampled subcarrier grid ===", flush=True)
        stage_n(epochs, best_model, best_cfg, seeds=(0,))

    print(f"\nALL STAGES DONE -> {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    main(epochs=args.epochs, smoke_test=args.smoke_test)
