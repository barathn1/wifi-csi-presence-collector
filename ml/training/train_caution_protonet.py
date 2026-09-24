"""CAUTION-style prototypical network + intruder-threshold trainer/evaluator (see
ml/models/prototypical_caution.py for the algorithm/architecture notes), run on the SAME pooled
channel-6 Day3-7 (2026-09-15/16/17/21/22, single-receiver-only) dataset and calibA preprocessing as
ml/training/train_day3_ch6_model.py, so results land in the same log
(ml/evaluation/results/day3_ch6_model_log.csv) and are directly comparable to that script's
whofi_transformer numbers for taskD_auth_vs_nonauth -- the log's fixed header has no separate `model`
column, so rows from this script are tagged "caution_protonet" in the `notes` field instead.

Two eval axes:
1. Open-set-by-stranger (`run_leave_one_unauthorized_out`): leave-one-unauthorized-person-out -- does
   the model reject a genuinely novel intruder identity it never saw during training/enrollment?
2. Open-set-by-day / cross-day (`run_leave_one_day_out`): leave-one-day-out -- does the model
   generalize to a new day's RF conditions? The harder, usually-unreported number per
   RESEARCH_NOTES.md sections 8/12.

Only authorized (anjali/barath) windows are ever used to train the encoder or fit prototypes --
unauthorized/none windows are used ONLY at evaluation time, matching the paper's "no prior knowledge
of intruders" design (section III-C): the intruder threshold T is calibrated from a held-out slice of
KNOWN (authorized) data only (the fraction of that held-out known data accepted at threshold T), never
from real intruder data.

    python3 -m ml.training.train_caution_protonet
    python3 -m ml.training.train_caution_protonet --episodes 200 --seed 1
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ml.data_pipeline.calibration import apply_variant_a, compute_day_baseline_from_sessions
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import leave_one_day_out, leave_one_unauthorized_person_out
from ml.data_pipeline.windowing import load_window
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.prototypical_caution import CautionEncoder, compute_prototypes, distance_ratio, prototypical_logits
from ml.training.train_day3_ch6_model import (
    CACHE_MODE,
    SINGLE_RECEIVER_MAC,
    build_day3_ch6_manifest,
    build_or_load_window_index,
    compute_dataset_target_rate_hz,
)

LOG_PATH = REPO_ROOT / "ml/evaluation/results/day3_ch6_model_log.csv"
LOG_FIELDNAMES = ["timestamp", "stage", "task", "held_out", "accuracy", "eer", "auroc",
                   "false_accept_unauthorized", "false_accept_none", "n_train", "n_test", "notes"]
CLASSES = ["anjali", "barath"]  # alphabetical, fixed prototype ordering
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_SUBCARRIERS = 128  # csi_resample.TARGET_SUBCARRIERS, matches train_day3_ch6_model.py
ENCODER_KWARGS = dict(d_model=32, n_heads=4, d_ff=64, num_layers=1, norm_first=False, dropout=0.2)
CALIB_HOLDOUT_FRAC = 0.2  # fraction of authorized TRAIN sessions reserved for threshold calibration
TARGET_ACCEPT = 0.95      # T is set so 95% of held-out KNOWN windows are accepted (paper's own
                          # "optimize T using only known-user data" philosophy -- section III-C)


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


class RowsWindowDataset(Dataset):
    """Minimal dataset over an explicit, already-filtered rows DataFrame -- sidesteps
    CsiWindowDataset's task-mask/reindex machinery, which this script's custom (auth-only-for-training,
    all-classes-for-eval) splits don't fit cleanly."""

    def __init__(self, rows: pd.DataFrame, calibration):
        self.rows = rows.reset_index(drop=True)
        self.calibration = calibration

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        amp, phase, _ = load_window(row)
        if self.calibration is not None:
            amp, phase = self.calibration(amp, phase, row)
        amp_t = torch.from_numpy(np.ascontiguousarray(amp, dtype=np.float32))
        phase_t = torch.from_numpy(np.ascontiguousarray(phase, dtype=np.float32))
        return amp_t, phase_t


@torch.no_grad()
def embed_rows(encoder: torch.nn.Module, rows: pd.DataFrame, calibration, batch_size: int = 64) -> np.ndarray:
    encoder.eval()
    loader = DataLoader(RowsWindowDataset(rows, calibration), batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = [encoder.embed(amp, phase).numpy() for amp, phase in loader]
    return np.concatenate(chunks, axis=0)


def load_batch(rows: pd.DataFrame, calibration) -> tuple[torch.Tensor, torch.Tensor]:
    amps, phases = [], []
    for _, row in rows.iterrows():
        amp, phase, _ = load_window(row)
        if calibration is not None:
            amp, phase = calibration(amp, phase, row)
        amps.append(amp)
        phases.append(phase)
    return (torch.from_numpy(np.ascontiguousarray(np.stack(amps), dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(np.stack(phases), dtype=np.float32)))


def sample_episode(rng: np.random.Generator, rows_by_class: dict[int, pd.DataFrame], n_support: int,
                    n_query: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    support_chunks, query_chunks, support_labels, query_labels = [], [], [], []
    for cls, rows in rows_by_class.items():
        n_take = n_support + n_query
        idx = rng.choice(len(rows), size=n_take, replace=len(rows) < n_take)
        chosen = rows.iloc[idx]
        support_chunks.append(chosen.iloc[:n_support])
        query_chunks.append(chosen.iloc[n_support:])
        support_labels += [cls] * n_support
        query_labels += [cls] * n_query
    return (pd.concat(support_chunks, ignore_index=True), np.array(support_labels),
            pd.concat(query_chunks, ignore_index=True), np.array(query_labels))


def train_protonet(rows_by_class: dict[int, pd.DataFrame], calibration, episodes: int, n_support: int,
                    n_query: int, seed: int, lr: float = 1e-3) -> CautionEncoder:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder = CautionEncoder(N_SUBCARRIERS, **ENCODER_KWARGS)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    n_classes = len(rows_by_class)
    encoder.train()
    for _ in range(episodes):
        support_rows, support_labels, query_rows, query_labels = sample_episode(rng, rows_by_class, n_support, n_query)
        s_amp, s_phase = load_batch(support_rows, calibration)
        q_amp, q_phase = load_batch(query_rows, calibration)
        support_emb = encoder.embed(s_amp, s_phase)
        query_emb = encoder.embed(q_amp, q_phase)
        prototypes = compute_prototypes(support_emb, torch.from_numpy(support_labels), n_classes)
        logits = prototypical_logits(query_emb, prototypes)
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(query_labels).long())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return encoder


def run_fold(train_rows: pd.DataFrame, test_rows: pd.DataFrame, calibration, episodes: int, n_support: int,
             n_query: int, seed: int) -> dict:
    auth_rows = train_rows[train_rows["label"] == "authorized"]
    sessions = sorted(auth_rows["session_dir"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sessions)
    n_calib = max(1, int(round(len(sessions) * CALIB_HOLDOUT_FRAC)))
    calib_sessions = set(shuffled[:n_calib])
    support_pool = auth_rows[~auth_rows["session_dir"].isin(calib_sessions)]
    calib_pool = auth_rows[auth_rows["session_dir"].isin(calib_sessions)]

    rows_by_class = {CLASS_TO_IDX[c]: support_pool[support_pool["person_id"] == c].reset_index(drop=True)
                      for c in CLASSES}
    min_available = min(len(r) for r in rows_by_class.values())
    assert min_available >= 2, f"too few authorized proto-support windows to run an episode: {min_available}"
    eff_support = max(1, min(n_support, min_available // 2))
    eff_query = max(1, min(n_query, min_available - eff_support))

    encoder = train_protonet(rows_by_class, calibration, episodes, eff_support, eff_query, seed)

    proto_embs, proto_labels = [], []
    for cls, rows in rows_by_class.items():
        proto_embs.append(embed_rows(encoder, rows, calibration))
        proto_labels += [cls] * len(rows)
    prototypes = compute_prototypes(
        torch.from_numpy(np.concatenate(proto_embs)), torch.from_numpy(np.array(proto_labels)), len(CLASSES))

    calib_emb = embed_rows(encoder, calib_pool, calibration)
    calib_R = distance_ratio(torch.from_numpy(calib_emb), prototypes).numpy()
    threshold = float(np.quantile(calib_R, TARGET_ACCEPT))

    test_emb = embed_rows(encoder, test_rows, calibration)
    test_R = distance_ratio(torch.from_numpy(test_emb), prototypes).numpy()
    y_true = (test_rows["label"].values == "authorized").astype(int)
    score = -test_R  # higher score = more confidently "known" (matches compute_eer/auroc's "higher=genuine")
    pred = (test_R <= threshold).astype(int)

    eer, _ = compute_eer(y_true, score)
    auroc = compute_auroc(y_true, score)
    out = {"accuracy": float((pred == y_true).mean()), "eer": eer, "auroc": auroc,
           "n_train": len(train_rows), "n_test": len(test_rows), "threshold": threshold}
    labels_arr = test_rows["label"].values
    for neg in ("unauthorized", "none"):
        m = labels_arr == neg
        if m.sum() > 0:
            out[f"false_accept_{neg}"] = float((pred[m] == 1).mean())
    return out


def run_leave_one_unauthorized_out(window_index: pd.DataFrame, calibration, episodes: int, n_support: int,
                                    n_query: int, seed: int) -> None:
    print(f"\n=== caution_protonet / taskD: leave-one-unauthorized-person-out ({len(window_index)} windows) ===")
    fold_metrics = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(window_index, include_none=True):
        train_rows, test_rows = window_index.iloc[train_idx], window_index.iloc[test_idx]
        metrics = run_fold(train_rows, test_rows, calibration, episodes, n_support, n_query, seed)
        fold_metrics.append(metrics)
        print(f"  held out '{held_out}': acc={metrics['accuracy']:.3f} auroc={metrics['auroc']:.3f} "
              f"eer={metrics['eer']:.3f} T={metrics['threshold']:.3f} "
              f"false_accept_unauth={metrics.get('false_accept_unauthorized', float('nan')):.3f} "
              f"false_accept_none={metrics.get('false_accept_none', float('nan')):.3f}")
        log_rows([{**metrics, "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "eval_loo",
                   "task": "taskD_auth_vs_nonauth", "held_out": held_out,
                   "notes": f"caution_protonet; leave-one-unauthorized-person-out; T={metrics['threshold']:.3f}"}])
    mean_auroc = np.nanmean([m["auroc"] for m in fold_metrics])
    mean_fa_unauth = np.nanmean([m.get("false_accept_unauthorized", float("nan")) for m in fold_metrics])
    mean_fa_none = np.nanmean([m.get("false_accept_none", float("nan")) for m in fold_metrics])
    print(f"  MEAN across {len(fold_metrics)} held-out strangers: auroc={mean_auroc:.3f} "
          f"false_accept_unauth={mean_fa_unauth:.3f} false_accept_none={mean_fa_none:.3f}")


def run_leave_one_day_out(window_index: pd.DataFrame, calibration, episodes: int, n_support: int,
                           n_query: int, seed: int) -> None:
    print(f"\n=== caution_protonet / taskD: leave-one-DAY-out, cross-day generalization "
          f"({len(window_index)} windows) ===")
    fold_metrics = []
    for held_out_date, train_idx, test_idx in leave_one_day_out(window_index):
        train_rows, test_rows = window_index.iloc[train_idx], window_index.iloc[test_idx]
        if test_rows["label"].nunique() < 2 and (test_rows["label"] == "authorized").sum() == 0:
            print(f"  skipping {held_out_date}: no authorized windows this day")
            continue
        metrics = run_fold(train_rows, test_rows, calibration, episodes, n_support, n_query, seed)
        fold_metrics.append(metrics)
        print(f"  held out day '{held_out_date}': acc={metrics['accuracy']:.3f} auroc={metrics['auroc']:.3f} "
              f"eer={metrics['eer']:.3f} T={metrics['threshold']:.3f} "
              f"false_accept_unauth={metrics.get('false_accept_unauthorized', float('nan')):.3f} "
              f"false_accept_none={metrics.get('false_accept_none', float('nan')):.3f}")
        log_rows([{**metrics, "timestamp": datetime.now(timezone.utc).isoformat(), "stage": "eval_day_loo",
                   "task": "taskD_auth_vs_nonauth", "held_out": held_out_date,
                   "notes": f"caution_protonet; leave-one-day-out (cross-day); T={metrics['threshold']:.3f}"}])
    mean_auroc = np.nanmean([m["auroc"] for m in fold_metrics])
    mean_fa_unauth = np.nanmean([m.get("false_accept_unauthorized", float("nan")) for m in fold_metrics])
    mean_fa_none = np.nanmean([m.get("false_accept_none", float("nan")) for m in fold_metrics])
    print(f"  MEAN across {len(fold_metrics)} held-out days: auroc={mean_auroc:.3f} "
          f"false_accept_unauth={mean_fa_unauth:.3f} false_accept_none={mean_fa_none:.3f}")


def main(episodes: int, n_support: int, n_query: int, seed: int) -> None:
    manifest = build_day3_ch6_manifest()
    target_rate_hz = compute_dataset_target_rate_hz(manifest)
    window_index = build_or_load_window_index(manifest, target_rate_hz)
    print(f"window index: {len(window_index)} windows across {manifest['session_dir'].nunique()} sessions")

    none_sessions = manifest.loc[manifest["label"] == "none", "session_dir"].tolist()
    baseline = compute_day_baseline_from_sessions(none_sessions, label="day3-7_ch6_protonet", mode=CACHE_MODE,
                                                   target_rate_hz=target_rate_hz, board_mac=SINGLE_RECEIVER_MAC)

    def calibration(amp, phase, row):
        return apply_variant_a(amp, phase, baseline)

    run_leave_one_unauthorized_out(window_index, calibration, episodes, n_support, n_query, seed)
    run_leave_one_day_out(window_index, calibration, episodes, n_support, n_query, seed)
    print(f"\nDone. Results appended to {LOG_PATH} (notes field tagged 'caution_protonet').")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--n-support", type=int, default=5)
    p.add_argument("--n-query", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(episodes=args.episodes, n_support=args.n_support, n_query=args.n_query, seed=args.seed)
