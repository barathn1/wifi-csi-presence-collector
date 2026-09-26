"""Same-day, session-disjoint sanity check: can the signature+EVM encoder tell identities apart
AT ALL on this project's single-ESP32 channel-6 data, with the cross-day drift question removed
entirely? Both `train_signature_evm_ch6_crossday.py` (zero-shot transfer to 2026-09-17) and
`recalibrate_signature_evm_ch6_day17.py` (same-day EVM refit) scored ~chance (AUROC 0.46-0.50) on
that unseen day. This isolates whether that's specifically a day-to-day generalization failure, or
whether the encoder can't separate these identities even within one day.

Trains AND tests entirely within 2026-09-17 (the richest single day: anjali 6 sessions, barath 4,
divya 3, manas 2, harshtiha 2), holding out each identity's LAST session as a session-disjoint test
set (never window-disjoint -- overlapping windows from a held-out session never appear in training).
Same architecture, same hyperparameters, same ENROLLED pair (anjali/barath) as the crossday script,
so the numbers below are directly comparable to that script's and the recalibration script's.

    python3 -m ml.training.sanity_check_signature_evm_sameday
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import numpy as np
import torch

from ml.data_pipeline.ablation_ch6_pipeline import build_ch6_manifest_all_labels, build_or_load_window_index, \
    compute_dataset_target_rate_hz
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import assert_no_group_leakage
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.signature_evm import evm_score, fit_evm
from ml.training.train_signature_evm_ch6_crossday import (
    ACCEPT_THRESHOLD, D_MODEL, ENROLLED, EVM_CENTROID_ONLY, EVM_TAU, RAW_CONFIG, SIGNATURE_DIM,
    TASK_NAME, embed_all, group_by_identity, train_encoder,
)

TARGET_DATE = "2026-09-17"
CHECKPOINT_PATH = REPO_ROOT / "ml/checkpoints/signature_evm_ch6_sameday_sanity.pt"
LOG_PATH = REPO_ROOT / "ml/evaluation/results/signature_evm_ch6_sameday_sanity_log.csv"
LOG_FIELDNAMES = ["timestamp", "stage", "identity", "auroc", "eer", "genuine_accept_rate",
                   "correct_identity_rate", "false_accept_rate", "n_train", "n_test", "notes"]
SEED = 0


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def main() -> None:
    print("rebuilding channel-6 manifest/window index (cached from prior runs)...", flush=True)
    manifest = build_ch6_manifest_all_labels()
    manifest = manifest[manifest["label"] != "none"].reset_index(drop=True)
    target_rate_hz = compute_dataset_target_rate_hz(manifest, RAW_CONFIG)
    window_index = build_or_load_window_index(manifest, RAW_CONFIG, target_rate_hz)

    day_df = window_index[window_index["date"] == TARGET_DATE].reset_index(drop=True)
    print(f"  {TARGET_DATE}: {len(day_df)} windows, "
          f"{day_df.groupby('person_id')['session_dir'].nunique().to_dict()} sessions/person", flush=True)

    test_session = {}
    for name in sorted(day_df["person_id"].unique()):
        sessions = sorted(day_df.loc[day_df["person_id"] == name, "session_dir"].unique())
        test_session[name] = sessions[-1]  # held-out LATEST session per identity, this project's
        # standard convention (person_home_only_experiment.py etc.)
    print(f"  held-out (test) session per identity: {test_session}", flush=True)

    is_test_session = day_df["session_dir"].isin(test_session.values()).values
    train_df = day_df[~is_test_session].reset_index(drop=True)
    test_df = day_df[is_test_session].reset_index(drop=True)
    assert not (set(train_df["session_dir"]) & set(test_df["session_dir"])), "session leakage"
    print(f"  train_windows={len(train_df)}  test_windows={len(test_df)} (session-disjoint, same day)",
          flush=True)

    dataset_train = CsiWindowDataset(train_df, TASK_NAME)
    dataset_test = CsiWindowDataset(test_df, TASK_NAME)
    n_subcarriers = dataset_train[0][0].shape[-1]
    print(f"train identities: {sorted(dataset_train.classes)}", flush=True)
    print(f"test identities:  {sorted(dataset_test.classes)}", flush=True)

    print("\n=== training signature encoder FROM SCRATCH, same day only "
          "(in-batch-negative contrastive loss) ===", flush=True)
    model = train_encoder(dataset_train, n_subcarriers, SEED)

    print("\n=== embedding train + held-out-session windows with the frozen encoder ===", flush=True)
    train_embeddings = embed_all(model, dataset_train)
    test_embeddings = embed_all(model, dataset_test)

    per_identity_train = group_by_identity(train_embeddings, dataset_train.y, np.arange(len(dataset_train)))
    evm = fit_evm(per_identity_train, tau=EVM_TAU, equalize=True, centroid_only=EVM_CENTROID_ONLY)

    psi = evm_score(evm, test_embeddings, ENROLLED)
    y_true = np.isin(dataset_test.y, ENROLLED).astype(int)
    accept_score = psi.max(axis=1)
    pred_identity = np.array(ENROLLED)[psi.argmax(axis=1)]

    auroc = compute_auroc(y_true, accept_score)
    eer, eer_thr = compute_eer(y_true, accept_score)
    genuine_mask = y_true == 1

    print(f"\n=== SAME-DAY SESSION-DISJOINT RESULT on {TARGET_DATE} "
          f"(n_test={len(test_df)}) ===", flush=True)
    print(f"  AUROC={auroc:.3f}  EER={eer:.3f}  <- threshold-free, the headline numbers here", flush=True)

    ts = datetime.now(timezone.utc).isoformat()
    log_rows_out = [{
        "timestamp": ts, "stage": "sameday_overall", "identity": "", "auroc": auroc, "eer": eer,
        "genuine_accept_rate": "", "correct_identity_rate": "", "false_accept_rate": "",
        "n_train": len(train_df), "n_test": len(test_df), "notes": f"held_out_sessions={test_session}",
    }]

    for label, thr, thr_desc in [(ACCEPT_THRESHOLD, ACCEPT_THRESHOLD, "carried-over fixed threshold"),
                                  (eer_thr, eer_thr, "EER threshold (hindsight-picked on this test set, "
                                                      "informational best-case only, not deployable)")]:
        accept_pred = accept_score >= thr
        genuine_accept_rate = float(accept_pred[genuine_mask].mean()) if genuine_mask.any() else float("nan")
        correct_identity_rate = float((pred_identity[genuine_mask] == dataset_test.y[genuine_mask]).mean()) \
            if genuine_mask.any() else float("nan")
        false_accept_rate = float(accept_pred[~genuine_mask].mean()) if (~genuine_mask).any() else float("nan")
        print(f"\n  @ threshold={thr:.3f} ({thr_desc}):", flush=True)
        print(f"    genuine_accept_rate={genuine_accept_rate:.1%}  "
              f"correct_identity_rate={correct_identity_rate:.1%}  "
              f"false_accept_rate(any stranger)={false_accept_rate:.1%}", flush=True)
        for name in ENROLLED:
            m = genuine_mask & (dataset_test.y == name)
            if not m.any():
                continue
            print(f"      {name:<10} n={int(m.sum()):>4}  accepted={float(accept_pred[m].mean()):.1%}  "
                  f"correctly-identified={float((pred_identity[m] == name).mean()):.1%}", flush=True)
        for name in sorted(set(dataset_test.y[~genuine_mask].tolist())):
            m = (~genuine_mask) & (dataset_test.y == name)
            print(f"      {name:<12} n={int(m.sum()):>4}  false_accept={float(accept_pred[m].mean()):.1%}",
                  flush=True)
        log_rows_out.append({
            "timestamp": ts, "stage": f"sameday_thr_{thr_desc.split()[0]}", "identity": "",
            "auroc": "", "eer": "", "genuine_accept_rate": genuine_accept_rate,
            "correct_identity_rate": correct_identity_rate, "false_accept_rate": false_accept_rate,
            "n_train": "", "n_test": len(test_df), "notes": f"threshold={thr:.3f} ({thr_desc})",
        })
    log_rows(log_rows_out)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "arch_kwargs": dict(n_subcarriers=n_subcarriers, d_model=D_MODEL, signature_dim=SIGNATURE_DIM),
        "evm": {name: evm[name] for name in ENROLLED},
        "enrolled": ENROLLED,
        "meta": {
            "purpose": "same-day session-disjoint sanity check, NOT a cross-day model",
            "date": TARGET_DATE, "held_out_sessions": test_session,
            "auroc": auroc, "eer": eer, "seed": SEED,
            "created": datetime.now(timezone.utc).isoformat(),
        },
    }, CHECKPOINT_PATH)
    print(f"\nsaved -> {CHECKPOINT_PATH}")
    print(f"log -> {LOG_PATH}")


if __name__ == "__main__":
    main()
