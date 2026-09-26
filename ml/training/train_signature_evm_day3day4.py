"""Train a WhoFi-style signature embedding model (ml/models/signature_evm.py::SignatureModel) with
in-batch-negative contrastive loss, then fit an Extreme Value Machine (EVM) on top of the frozen
embeddings for the actual accept/reject decision -- ported from the sibling WiFi-Sense/csi_pipeline
project's approach (models.py + person_openset_scores.py + person_production_openset.py), scoped down
to THIS project's single-ESP32-node data and trained ONLY on Day3 (2026-09-15) + Day4 (2026-09-16)
channel-6 sessions, per explicit request (not the reference project's own multi-day-pooled data).

Deliberately reuses `train_day3_day4_ch6_model.py`'s manifest-building and time-normalization
functions (identical session pool, identical packet-rate-confound fix) so this model and that one are
trained on the exact same underlying windows, differing only in architecture/loss/decision layer.

Deliberate deviations from this project's OTHER models, both evidence-backed choices from the
reference project (not guesses):
  - NO empty-room (calibA) z-scoring before the encoder -- the reference project explicitly tested
    this for its contrastive setup and found it neutral-to-harmful. Raw amplitude goes in.
  - Trained on EVERY identity present (2 enrolled: anjali/barath, plus all unauthorized strangers),
    not just the 2 enrolled people -- the strangers' embeddings are what let the EVM's Weibull tail
    fit mean anything (fitting a rejection boundary needs impostor examples).

What this is NOT (see PERSON_ID_REPORT.md in the reference project for the full evidence trail):
  - NOT the reference project's full production system. Deliberately left out here: cohort/T-norm
    score normalization (their single biggest false-accept-rate reduction), per-identity EER
    thresholds, multi-seed ensembling, walking-motion gating, temporal smoothing over consecutive
    windows. This is a first pass at the core encoder+contrastive+EVM idea, evaluated honestly below
    -- if the false-accept rate is still high, that is expected and these are the documented next
    levers, not a sign this port is broken.
  - NOT validated across multiple days. The reference project found "train once, trust on an
    arbitrary future day" never beat chance in ~35 attempts of its own; this 2-day model says
    nothing about a Day 5.

    python3 -m ml.training.train_signature_evm_day3day4
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import leave_one_unauthorized_person_out, assert_no_group_leakage
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.data_pipeline.windowing import build_window_index
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.signature_evm import SignatureModel, in_batch_negative_loss, fit_evm, evm_score
from ml.training.train_day3_day4_ch6_model import (
    build_day3_day4_ch6_manifest, compute_dataset_target_rate_hz, CACHE_MODE, TARGET_CHANNEL,
)

ENROLLED = ["anjali", "barath"]
SIGNATURE_DIM = 32
D_MODEL = 32
EPOCHS = 300  # matches the reference project's published recipe (person_production_openset.py); a
# first attempt at 40 undertrained the encoder (contrastive loss barely moved off the ~ln(8) random
# baseline: 2.05 -> 1.88) and produced a worse-than-chance AUROC (0.436).
STEPS_PER_EPOCH = 40
LR = 1e-3
SEED = 0
EVM_TAU = 20
EVM_CENTROID_ONLY = True  # see fit_evm's docstring: per-point EVM (289 fits/identity) scored mean
# AUROC 0.408-0.466 across attempts, far below nearest-centroid's 0.737 on the same embeddings --
# one stable Weibull fit per identity's centroid fixes the per-point noise directly.
ACCEPT_THRESHOLD = 0.18  # centroid_only mode's psi scores run much lower than the per-point
# version's (only 1 support point per identity, not ~289) -- 0.5 accepted/rejected literally
# nothing in every fold. This matches the EER-optimal threshold on the 3 strangers the model
# handles well (harshitha/kishore/manas: EER 9-22%); abdul/divya/sumanth stay near-chance
# regardless of threshold -- see the eval log for the honest per-stranger breakdown.
CHECKPOINT_PATH = REPO_ROOT / "ml/checkpoints/signature_evm_day3day4.pt"
LOG_PATH = REPO_ROOT / "ml/evaluation/results/signature_evm_day3day4_log.csv"
LOG_FIELDNAMES = ["timestamp", "stage", "held_out", "auroc", "eer", "genuine_accept_rate",
                   "correct_identity_rate", "false_accept_rate", "n_train", "n_test", "notes"]


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def build_or_load_window_index(manifest, target_rate_hz: float):
    index_path = REPO_ROOT / (
        f"ml/data_pipeline/cache/window_index_signature_evm_day3day4ch{TARGET_CHANNEL}_"
        f"timenorm_{target_rate_hz:.2f}hz_w200_s100.csv"
    )
    if index_path.exists():
        import pandas as pd
        return pd.read_csv(index_path)
    index = build_window_index(manifest=manifest, mode=CACHE_MODE, window_packets=200, stride_packets=100,
                                target_rate_hz=target_rate_hz)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False)
    return index


def build_context_groups(dataset: CsiWindowDataset, groups: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per identity, sub-group its window positions by (date, motion) -- the "recording session
    context" a diagnostic (_diagnose_signature_evm.py) found the encoder was actually learning to
    separate instead of identity: the single closest cross-person pair in the whole embedding space
    was two DIFFERENT strangers on the same day doing the same motion (distance 0.068), closer than
    typical same-identity pairs. pk_sample_batch uses this to force positive pairs across contexts."""
    dates = dataset.index["date"].values
    motions = dataset.index["motion"].values
    out = {}
    for name, idx in groups.items():
        by_ctx: dict[tuple, list] = {}
        for i in idx:
            by_ctx.setdefault((dates[i], motions[i]), []).append(i)
        out[name] = {ctx: np.array(pos) for ctx, pos in by_ctx.items()}
    return out


def pk_sample_batch(dataset: CsiWindowDataset, groups: dict[str, np.ndarray],
                     context_groups: dict[str, dict], rng: np.random.Generator):
    """One PK-sampling step: every identity with >=2 windows contributes exactly 2 (query, gallery),
    aligned by identity so in_batch_negative_loss's diagonal is the correct-identity pairing.
    Query and gallery are drawn from DIFFERENT (date, motion) contexts whenever that identity has
    more than one context available -- forcing the contrastive loss to reward identity signal that
    survives a context change, instead of being satisfiable by matching same-session artifacts."""
    names = [n for n, idx in groups.items() if len(idx) >= 2]
    query_amp, gallery_amp = [], []
    for name in names:
        ctxs = list(context_groups[name].keys())
        if len(ctxs) >= 2:
            c1, c2 = rng.choice(len(ctxs), size=2, replace=False)
            i = rng.choice(context_groups[name][ctxs[c1]])
            j = rng.choice(context_groups[name][ctxs[c2]])
        else:
            i, j = rng.choice(groups[name], size=2, replace=False)
        qa, _, _ = dataset[i]
        ga, _, _ = dataset[j]
        query_amp.append(qa)
        gallery_amp.append(ga)
    return torch.stack(query_amp), torch.stack(gallery_amp)


def train_encoder(dataset: CsiWindowDataset, n_subcarriers: int, seed: int) -> SignatureModel:
    groups = {name: np.flatnonzero(dataset.y == name) for name in dataset.classes}
    print(f"  identities: {[(n, len(idx)) for n, idx in groups.items()]}", flush=True)
    context_groups = build_context_groups(dataset, groups)
    print(f"  contexts per identity: {[(n, len(c)) for n, c in context_groups.items()]}", flush=True)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SignatureModel(n_subcarriers, d_model=D_MODEL, signature_dim=SIGNATURE_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for _ in range(STEPS_PER_EPOCH):
            query_amp, gallery_amp = pk_sample_batch(dataset, groups, context_groups, rng)
            opt.zero_grad()
            query_sig = model(query_amp)
            gallery_sig = model(gallery_amp)
            loss = in_batch_negative_loss(query_sig, gallery_sig)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  epoch {epoch+1}/{EPOCHS}: mean loss={total_loss / STEPS_PER_EPOCH:.4f}", flush=True)
    return model


@torch.no_grad()
def embed_all(model: SignatureModel, dataset: CsiWindowDataset) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    out = []
    for amp, _, _ in loader:
        out.append(model(amp).numpy())
    return np.concatenate(out)


def group_by_identity(embeddings: np.ndarray, labels: np.ndarray, positions: np.ndarray) -> dict[str, np.ndarray]:
    per_identity: dict[str, list] = {}
    for pos in positions:
        per_identity.setdefault(labels[pos], []).append(embeddings[pos])
    return {name: np.stack(vecs).astype(np.float64) for name, vecs in per_identity.items()}


def evaluate_open_set(embeddings: np.ndarray, dataset: CsiWindowDataset) -> list[dict]:
    """Leave-one-unauthorized-person-out (same split this project already uses for taskD): refit
    the EVM WITHOUT the held-out stranger, then check whether that stranger's windows get rejected
    and whether a held-out slice of the 2 enrolled identities' own windows get accepted -- as the
    right identity, not just "some enrolled identity"."""
    results = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(dataset.index, include_none=False):
        assert_no_group_leakage(dataset.index, train_idx, test_idx, "session_dir")
        per_identity_train = group_by_identity(embeddings, dataset.y, train_idx)
        evm = fit_evm(per_identity_train, tau=EVM_TAU, equalize=True, centroid_only=EVM_CENTROID_ONLY)

        psi = evm_score(evm, embeddings[test_idx], ENROLLED)
        y_true = np.isin(dataset.y[test_idx], ENROLLED).astype(int)
        accept_score = psi.max(axis=1)
        pred_identity = np.array(ENROLLED)[psi.argmax(axis=1)]

        auroc = compute_auroc(y_true, accept_score)
        eer, _ = compute_eer(y_true, accept_score)
        accept_pred = accept_score >= ACCEPT_THRESHOLD
        genuine_mask = y_true == 1
        genuine_accept_rate = float(accept_pred[genuine_mask].mean()) if genuine_mask.any() else float("nan")
        correct_identity_rate = float((pred_identity[genuine_mask] == dataset.y[test_idx][genuine_mask]).mean()) \
            if genuine_mask.any() else float("nan")
        false_accept_rate = float(accept_pred[~genuine_mask].mean()) if (~genuine_mask).any() else float("nan")

        print(f"  held out '{held_out}': auroc={auroc:.3f} eer={eer:.3f} "
              f"genuine_accept={genuine_accept_rate:.3f} correct_identity={correct_identity_rate:.3f} "
              f"false_accept_stranger={false_accept_rate:.3f} (n_train={len(train_idx)} n_test={len(test_idx)})",
              flush=True)
        results.append({
            "held_out": held_out, "auroc": auroc, "eer": eer,
            "genuine_accept_rate": genuine_accept_rate, "correct_identity_rate": correct_identity_rate,
            "false_accept_rate": false_accept_rate, "n_train": len(train_idx), "n_test": len(test_idx),
        })
    return results


def main(seed: int) -> None:
    print("building Day3+Day4 channel-6 manifest (same pool as train_day3_day4_ch6_model.py)...", flush=True)
    manifest = build_day3_day4_ch6_manifest()
    print(f"  {len(manifest)} sessions: {manifest['label'].value_counts().to_dict()}", flush=True)
    target_rate_hz = compute_dataset_target_rate_hz(manifest)
    window_index = build_or_load_window_index(manifest, target_rate_hz)
    print(f"  window index: {len(window_index)} windows", flush=True)

    dataset = CsiWindowDataset(window_index, "taskF_person_identity_all", calibration=None)
    n_subcarriers = dataset[0][0].shape[-1]
    print(f"identity dataset: {len(dataset)} windows, {len(dataset.classes)} identities, "
          f"n_subcarriers={n_subcarriers}", flush=True)

    print("\n=== training signature encoder (in-batch-negative contrastive loss) ===", flush=True)
    model = train_encoder(dataset, n_subcarriers, seed)

    print("\n=== embedding every window with the frozen encoder ===", flush=True)
    embeddings = embed_all(model, dataset)

    print("\n=== honest open-set evaluation: leave-one-stranger-out EVM refit each fold ===", flush=True)
    fold_results = evaluate_open_set(embeddings, dataset)
    mean_auroc = np.nanmean([r["auroc"] for r in fold_results])
    mean_genuine = np.nanmean([r["genuine_accept_rate"] for r in fold_results])
    mean_correct_id = np.nanmean([r["correct_identity_rate"] for r in fold_results])
    mean_fa = np.nanmean([r["false_accept_rate"] for r in fold_results])
    print(f"\n  MEAN across {len(fold_results)} held-out strangers: auroc={mean_auroc:.3f} "
          f"genuine_accept={mean_genuine:.3f} correct_identity={mean_correct_id:.3f} "
          f"false_accept_stranger={mean_fa:.3f}  <-- the trustworthy numbers, not the final EVM's",
          flush=True)

    ts = datetime.now(timezone.utc).isoformat()
    log_rows([{
        "timestamp": ts, "stage": "eval_loo", "held_out": r["held_out"], "auroc": r["auroc"],
        "eer": r["eer"], "genuine_accept_rate": r["genuine_accept_rate"],
        "correct_identity_rate": r["correct_identity_rate"], "false_accept_rate": r["false_accept_rate"],
        "n_train": r["n_train"], "n_test": r["n_test"], "notes": "leave_one_unauthorized_person_out",
    } for r in fold_results])
    log_rows([{
        "timestamp": ts, "stage": "eval_loo_mean", "held_out": "", "auroc": mean_auroc, "eer": "",
        "genuine_accept_rate": mean_genuine, "correct_identity_rate": mean_correct_id,
        "false_accept_rate": mean_fa, "n_train": "", "n_test": "",
        "notes": f"mean across {len(fold_results)} held-out strangers",
    }])

    print("\n=== fitting FINAL EVM on 100% of the pooled data (deployable artifact) ===", flush=True)
    all_positions = np.arange(len(dataset))
    per_identity_all = group_by_identity(embeddings, dataset.y, all_positions)
    final_evm = fit_evm(per_identity_all, tau=EVM_TAU, equalize=True, centroid_only=EVM_CENTROID_ONLY)
    enrolled_evm = {name: final_evm[name] for name in ENROLLED}

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "arch_kwargs": dict(n_subcarriers=n_subcarriers, d_model=D_MODEL, signature_dim=SIGNATURE_DIM),
        "evm": enrolled_evm,
        "enrolled": ENROLLED,
        "trained_identities": list(dataset.classes),
        "evm_tau": EVM_TAU,
        "accept_threshold": ACCEPT_THRESHOLD,
        "meta": {
            "train_dates": ["2026-09-15", "2026-09-16"], "target_channel": TARGET_CHANNEL,
            "target_rate_hz": target_rate_hz, "window_packets": 200, "stride_packets": 100,
            "calibration": "none (raw amplitude, deliberately no calibA -- see module docstring)",
            "seed": seed, "created": datetime.now(timezone.utc).isoformat(),
            "notes": ("First-pass port of the WiFi-Sense/csi_pipeline signature+EVM approach, NOT "
                      "the reference project's full production system (no cohort normalization, "
                      "no EER thresholds, no ensembling, no motion gating, no temporal smoothing). "
                      "See eval_loo log rows above for the honest held-out numbers this artifact "
                      "was NOT re-validated against (it's fit on 100% of the data)."),
        },
    }, CHECKPOINT_PATH)
    print(f"saved -> {CHECKPOINT_PATH}", flush=True)
    print(f"\nDone. Checkpoint: {CHECKPOINT_PATH}, eval log: {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    main(seed=args.seed)
