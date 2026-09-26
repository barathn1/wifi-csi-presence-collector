"""Same-day recalibration for the cross-day signature+EVM model (`train_signature_evm_ch6_crossday.py`),
which found zero-shot transfer to 2026-09-17 essentially at chance (AUROC 0.460). This follows the one
lever the reference WiFi-Sense project's home-deployment review documented as actually working: freeze
the trained encoder, refit ONLY the EVM reference points (+ implicitly the threshold) using a small
slice of the new day's own data, then test honestly on the rest of that day.

Split (session-disjoint, never window-disjoint -- overlapping windows from the same session must never
land on both sides): for each ENROLLED identity (anjali, barath), their EARLIEST 2026-09-17 session is
the "recalibration" slice (mimics recalibrating first thing when arriving that day); every later session
of theirs that day is the honest test set. The three Sept-17 strangers (divya, manas, harshtiha -- the
last never seen at all before this day) are NEVER used for recalibration -- 100% of their windows stay
in the test set, so the false-accept numbers below are not inflated by leaking any of their own data in.

The EVM's impostor ("others") pool for anjali/barath's refit reuses the ALREADY-COMPUTED Day3+Day4
stranger embeddings from the frozen encoder (no need to touch Sept-17 stranger data for this) --
importantly this also includes barath's OWN Day3+Day4 embeddings as part of anjali's impostor pool and
vice versa, which is what actually stresses the "reject the other enrolled identity" cohort-confusion
case the reference project's Key Finding 1 was about.

    python3 -m ml.training.recalibrate_signature_evm_ch6_day17
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import numpy as np
import torch

from ml.data_pipeline.ablation_ch6_pipeline import build_ch6_manifest_all_labels, build_or_load_window_index, \
    compute_dataset_target_rate_hz
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.signature_evm import SignatureModel, evm_score, fit_evm
from ml.training.train_signature_evm_ch6_crossday import (
    ACCEPT_THRESHOLD, ENROLLED, EVM_CENTROID_ONLY, EVM_TAU, RAW_CONFIG, TASK_NAME,
    embed_all, group_by_identity,
)

TEST_DATE = "2026-09-17"
TRAIN_DATES = ["2026-09-15", "2026-09-16"]
SOURCE_CHECKPOINT = REPO_ROOT / "ml/checkpoints/signature_evm_ch6_crossday.pt"
OUT_CHECKPOINT = REPO_ROOT / "ml/checkpoints/signature_evm_ch6_day17_recalibrated.pt"
LOG_PATH = REPO_ROOT / "ml/evaluation/results/signature_evm_ch6_day17_recal_log.csv"
LOG_FIELDNAMES = ["timestamp", "stage", "identity", "auroc", "eer", "genuine_accept_rate",
                   "correct_identity_rate", "false_accept_rate", "n_recal", "n_test", "notes"]


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def pick_recal_sessions(window_index) -> dict[str, str]:
    """Earliest 2026-09-17 session per enrolled identity, by session_dir's own timestamp-sortable
    name (`.../20260917_HHMMSS_person`) -- sorted() is chronological here, same convention already
    used elsewhere in this project (`sorted(files)[...]`)."""
    day_df = window_index[window_index["date"] == TEST_DATE]
    recal_session = {}
    for name in ENROLLED:
        sessions = sorted(day_df.loc[day_df["person_id"] == name, "session_dir"].unique())
        assert sessions, f"no {TEST_DATE} sessions found for enrolled identity {name!r}"
        recal_session[name] = sessions[0]
    return recal_session


def main() -> None:
    print("rebuilding channel-6 manifest/window index (cached from the crossday run)...", flush=True)
    manifest = build_ch6_manifest_all_labels()
    manifest = manifest[manifest["label"] != "none"].reset_index(drop=True)
    target_rate_hz = compute_dataset_target_rate_hz(manifest, RAW_CONFIG)
    window_index = build_or_load_window_index(manifest, RAW_CONFIG, target_rate_hz)
    print(f"  window index: {len(window_index)} windows", flush=True)

    print(f"\nloading FROZEN encoder from {SOURCE_CHECKPOINT} (no retraining)...", flush=True)
    ck = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    model = SignatureModel(**ck["arch_kwargs"])
    model.load_state_dict(ck["model_state"])
    model.eval()

    dataset_all = CsiWindowDataset(window_index, TASK_NAME)
    print(f"embedding all {len(dataset_all)} windows once with the frozen encoder...", flush=True)
    embeddings = embed_all(model, dataset_all)
    idx_all = dataset_all.index

    recal_session = pick_recal_sessions(idx_all)
    print(f"\nrecalibration sessions (earliest {TEST_DATE} session per enrolled identity):", flush=True)
    for name, sess in recal_session.items():
        print(f"  {name}: {sess}", flush=True)

    is_day17 = (idx_all["date"] == TEST_DATE).values
    is_recal_session = idx_all["session_dir"].isin(recal_session.values()).values
    recal_mask = is_day17 & is_recal_session
    test_mask = is_day17 & ~is_recal_session
    recal_pos = np.flatnonzero(recal_mask)
    test_pos = np.flatnonzero(test_mask)
    print(f"\n{TEST_DATE} split: recal_windows={len(recal_pos)} (enrolled only) "
          f"test_windows={len(test_pos)} (everyone, incl. every stranger's full session set)", flush=True)
    assert not (set(idx_all.iloc[recal_pos]["session_dir"]) & set(idx_all.iloc[test_pos]["session_dir"])), \
        "leakage: a session appears on both sides of the recal/test split"

    per_identity_recal: dict[str, np.ndarray] = {}
    for name in ENROLLED:
        m = recal_mask & (idx_all["person_id"] == name).values
        per_identity_recal[name] = embeddings[m]
        print(f"  {name}: {m.sum()} fresh {TEST_DATE} windows for recalibration", flush=True)

    train_stranger_mask = idx_all["date"].isin(TRAIN_DATES).values & ~idx_all["person_id"].isin(ENROLLED).values
    for name in sorted(set(idx_all.loc[train_stranger_mask, "person_id"])):
        m = train_stranger_mask & (idx_all["person_id"] == name).values
        per_identity_recal[name] = embeddings[m]

    print(f"\nrefitting EVM: enrolled identities' own points = fresh {TEST_DATE} recal windows; "
          f"impostor pool = {TRAIN_DATES} stranger windows (unchanged from the crossday run)...", flush=True)
    evm = fit_evm(per_identity_recal, tau=EVM_TAU, equalize=True, centroid_only=EVM_CENTROID_ONLY)

    test_y = idx_all.iloc[test_pos]["person_id"].values
    test_emb = embeddings[test_pos]
    psi = evm_score(evm, test_emb, ENROLLED)
    y_true = np.isin(test_y, ENROLLED).astype(int)
    accept_score = psi.max(axis=1)
    pred_identity = np.array(ENROLLED)[psi.argmax(axis=1)]

    auroc = compute_auroc(y_true, accept_score)
    eer, eer_thr = compute_eer(y_true, accept_score)
    accept_pred = accept_score >= ACCEPT_THRESHOLD
    genuine_mask = y_true == 1
    genuine_accept_rate = float(accept_pred[genuine_mask].mean()) if genuine_mask.any() else float("nan")
    correct_identity_rate = float((pred_identity[genuine_mask] == test_y[genuine_mask]).mean()) \
        if genuine_mask.any() else float("nan")
    false_accept_rate = float(accept_pred[~genuine_mask].mean()) if (~genuine_mask).any() else float("nan")

    print(f"\n=== RECALIBRATED RESULT on {TEST_DATE} (test={len(test_pos)} windows, "
          f"recal={len(recal_pos)} windows) ===", flush=True)
    print(f"  AUROC={auroc:.3f}  EER={eer:.3f} (thr={eer_thr:.3f})  "
          f"genuine_accept_rate={genuine_accept_rate:.1%}  correct_identity_rate={correct_identity_rate:.1%}  "
          f"false_accept_rate(any stranger)={false_accept_rate:.1%}  (accept_threshold={ACCEPT_THRESHOLD})",
          flush=True)

    ts = datetime.now(timezone.utc).isoformat()
    log_rows_out = [{
        "timestamp": ts, "stage": "recal_overall", "identity": "", "auroc": auroc, "eer": eer,
        "genuine_accept_rate": genuine_accept_rate, "correct_identity_rate": correct_identity_rate,
        "false_accept_rate": false_accept_rate, "n_recal": len(recal_pos), "n_test": len(test_pos),
        "notes": f"recal_sessions={recal_session}",
    }]

    print(f"\n  per-enrolled-identity genuine accept rate on {TEST_DATE} (recal-holdout sessions only):",
          flush=True)
    for name in ENROLLED:
        m = genuine_mask & (test_y == name)
        if not m.any():
            continue
        rate = float(accept_pred[m].mean())
        correct = float((pred_identity[m] == name).mean())
        print(f"    {name:<10} n={int(m.sum()):>4}  accepted={rate:.1%}  correctly-identified={correct:.1%}",
              flush=True)
        log_rows_out.append({
            "timestamp": ts, "stage": "recal_per_enrolled", "identity": name, "auroc": "", "eer": "",
            "genuine_accept_rate": rate, "correct_identity_rate": correct, "false_accept_rate": "",
            "n_recal": "", "n_test": int(m.sum()), "notes": TEST_DATE,
        })

    print(f"\n  per-stranger false-accept rate on {TEST_DATE} (0% = correctly rejected every time):",
          flush=True)
    for name in sorted(set(test_y[~genuine_mask].tolist())):
        m = (~genuine_mask) & (test_y == name)
        fa = float(accept_pred[m].mean())
        novel = " <-- NEVER SEEN BEFORE (no recal or train data at all)" if name not in per_identity_recal else ""
        print(f"    {name:<12} n={int(m.sum()):>4}  false_accept={fa:.1%}{novel}", flush=True)
        log_rows_out.append({
            "timestamp": ts, "stage": "recal_per_stranger", "identity": name, "auroc": "", "eer": "",
            "genuine_accept_rate": "", "correct_identity_rate": "", "false_accept_rate": fa,
            "n_recal": "", "n_test": int(m.sum()), "notes": "novel" if novel else "seen_in_train",
        })
    log_rows(log_rows_out)

    OUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "arch_kwargs": ck["arch_kwargs"],
        "evm": {name: evm[name] for name in ENROLLED},
        "enrolled": ENROLLED,
        "evm_tau": EVM_TAU,
        "accept_threshold": ACCEPT_THRESHOLD,
        "meta": {
            "base_checkpoint": str(SOURCE_CHECKPOINT), "recal_date": TEST_DATE,
            "recal_sessions": recal_session, "impostor_pool_dates": TRAIN_DATES,
            "recal_auroc": auroc, "recal_eer": eer,
            "recal_genuine_accept_rate": genuine_accept_rate,
            "recal_correct_identity_rate": correct_identity_rate,
            "recal_false_accept_rate": false_accept_rate,
            "created": datetime.now(timezone.utc).isoformat(),
            "notes": ("Encoder frozen from signature_evm_ch6_crossday.pt (trained on 2026-09-15/16 "
                      "only). Only the EVM's enrolled reference points were refit, using each "
                      "enrolled identity's own earliest 2026-09-17 session. Evaluated on that "
                      "identity's LATER 2026-09-17 sessions plus every 2026-09-17 stranger session "
                      "(none of which were used for recalibration)."),
        },
    }, OUT_CHECKPOINT)
    print(f"\nsaved -> {OUT_CHECKPOINT}")
    print(f"log -> {LOG_PATH}")


if __name__ == "__main__":
    main()
