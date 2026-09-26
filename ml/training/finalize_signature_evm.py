"""Refit the EVM decision layer on an ALREADY-TRAINED signature encoder checkpoint, without
retraining the encoder itself -- useful because the encoder (300 epochs) is the expensive part,
while refitting the EVM given fixed embeddings is seconds. Exists because iterating on the decision
layer alone (tau, centroid_only, accept threshold) doesn't require repaying the encoder's cost every
time; see train_signature_evm_day3day4.py's module docstring for the full investigation this
followed (per-point EVM scored mean AUROC 0.408-0.466 across several encoder retrains; switching to
`fit_evm(..., centroid_only=True)` on the SAME embeddings jumped that to 0.713, matching a plain
nearest-centroid sanity check's 0.737 -- confirming the encoder was fine all along and the per-point
EVM machinery was the broken piece).

    python3 -m ml.training.finalize_signature_evm
"""
from __future__ import annotations

import numpy as np
import torch

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.models.signature_evm import SignatureModel
from ml.training.train_day3_day4_ch6_model import build_day3_day4_ch6_manifest, compute_dataset_target_rate_hz
from ml.training.train_signature_evm_day3day4 import (
    ACCEPT_THRESHOLD, CHECKPOINT_PATH, ENROLLED, EVM_CENTROID_ONLY, EVM_TAU,
    build_or_load_window_index, embed_all, evaluate_open_set, group_by_identity, log_rows,
)
from datetime import datetime, timezone
from ml.models.signature_evm import fit_evm


def main() -> None:
    print(f"loading existing checkpoint (encoder weights unchanged): {CHECKPOINT_PATH}", flush=True)
    ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = SignatureModel(**ck["arch_kwargs"])
    model.load_state_dict(ck["model_state"])
    model.eval()

    manifest = build_day3_day4_ch6_manifest()
    target_rate_hz = compute_dataset_target_rate_hz(manifest)
    window_index = build_or_load_window_index(manifest, target_rate_hz)
    dataset = CsiWindowDataset(window_index, "taskF_person_identity_all", calibration=None)
    embeddings = embed_all(model, dataset)
    print(f"  {len(dataset)} windows embedded", flush=True)

    print(f"\n=== honest open-set evaluation (centroid_only={EVM_CENTROID_ONLY}, "
          f"accept_threshold={ACCEPT_THRESHOLD}) ===", flush=True)
    fold_results = evaluate_open_set(embeddings, dataset)
    mean_auroc = np.nanmean([r["auroc"] for r in fold_results])
    mean_genuine = np.nanmean([r["genuine_accept_rate"] for r in fold_results])
    mean_correct_id = np.nanmean([r["correct_identity_rate"] for r in fold_results])
    mean_fa = np.nanmean([r["false_accept_rate"] for r in fold_results])
    print(f"\n  MEAN across {len(fold_results)} held-out strangers: auroc={mean_auroc:.3f} "
          f"genuine_accept={mean_genuine:.3f} correct_identity={mean_correct_id:.3f} "
          f"false_accept_stranger={mean_fa:.3f}", flush=True)

    ts = datetime.now(timezone.utc).isoformat()
    log_rows([{
        "timestamp": ts, "stage": "eval_loo_centroid_refit", "held_out": r["held_out"], "auroc": r["auroc"],
        "eer": r["eer"], "genuine_accept_rate": r["genuine_accept_rate"],
        "correct_identity_rate": r["correct_identity_rate"], "false_accept_rate": r["false_accept_rate"],
        "n_train": r["n_train"], "n_test": r["n_test"],
        "notes": f"centroid_only EVM refit, accept_threshold={ACCEPT_THRESHOLD}, same encoder checkpoint",
    } for r in fold_results])

    print("\n=== refitting FINAL centroid-only EVM on 100% of the pooled data ===", flush=True)
    all_positions = np.arange(len(dataset))
    per_identity_all = group_by_identity(embeddings, dataset.y, all_positions)
    final_evm = fit_evm(per_identity_all, tau=EVM_TAU, equalize=True, centroid_only=EVM_CENTROID_ONLY)
    enrolled_evm = {name: final_evm[name] for name in ENROLLED}

    ck["evm"] = enrolled_evm
    ck["accept_threshold"] = ACCEPT_THRESHOLD
    ck["meta"]["centroid_only"] = EVM_CENTROID_ONLY
    ck["meta"]["notes"] += (" | EVM refit via finalize_signature_evm.py: switched to "
                             "centroid_only=True (one Weibull fit per identity's centroid instead "
                             "of ~289 per-point fits) after a diagnostic showed the per-point "
                             "version scored far below a plain nearest-centroid baseline on the "
                             "same embeddings.")
    torch.save(ck, CHECKPOINT_PATH)
    print(f"saved -> {CHECKPOINT_PATH}", flush=True)


if __name__ == "__main__":
    main()
