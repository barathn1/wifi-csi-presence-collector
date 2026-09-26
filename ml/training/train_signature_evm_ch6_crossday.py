"""Cross-day accuracy check for the ported WiFi-Sense signature+EVM ("home model") on THIS
project's single-ESP32-node channel-6 data, across all three available dates: 2026-09-15 (Day3),
2026-09-16 (Day4), 2026-09-17 (Day5).

`train_signature_evm_day3day4.py` trained this same model on Day3+Day4 only and explicitly flagged
that it says "nothing about a Day 5" (never evaluated on an unseen future day). This script is that
missing evaluation: train on Day3+Day4 (`splits.day_disjoint_split`'s "all but last date"), then
score Day5 -- entirely unseen during training/EVM-fitting -- for genuine cross-day generalization,
not a same-pool holdout.

Model/architecture is unchanged from the reference (`ml/models/signature_evm.py`): this project's
data is already single-ESP32-node, so there is no MultiBranchEncoder/node-fusion to remove -- the
reference project's `BranchEncoder` (one transformer branch) already matches a single sensing node
one-for-one.

Manifest/preprocessing reuses `ablation_ch6_pipeline`'s validated session handling (not
`train_day3_day4_ch6_model.py`, which no longer exists in this checkout) because it already fixes
two things a from-scratch rebuild would otherwise silently miss:
  - the one 2026-09-15 session captured on channel 11 (a mid-collection router hop), correctly
    excluded rather than lumped in as channel 6.
  - `select_dominant_packets`'s MAC/PHY-config validation, so a session's CSI is always the one
    real sensing link at one consistent PHY config, not an accidental mix.
A `PreprocessConfig` with every stage off (no outlier removal, no smoothing) is used deliberately --
matching the reference project's own finding for this contrastive setup ("raw amplitude in", no
empty-room calibration) -- the only unavoidable step is per-session time-resampling onto one common
packet rate, needed so every window is the same shape across sessions with different native rates.

Trained on every identity present in Day3+Day4 (2 enrolled: anjali/barath, plus every unauthorized
stranger seen those two days) -- the strangers are what let the EVM's Weibull tail fit mean
anything at all (fitting a rejection boundary needs impostor examples). Day5 also has one
unauthorized identity never seen in training (a spelling variant of an otherwise-seen person) --
left in deliberately as the honest "truly novel stranger" case that matters most for the FAR number,
not filtered out because it's inconvenient.

    python3 -m ml.training.train_signature_evm_ch6_crossday
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data_pipeline.ablation_ch6_pipeline import (
    build_ch6_manifest_all_labels, build_or_load_window_index, compute_dataset_target_rate_hz,
)
from ml.data_pipeline.ablation_preprocessing import PreprocessConfig
from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import assert_no_group_leakage, day_disjoint_split
from ml.data_pipeline.tasks import TASKS
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.evaluation.metrics import compute_auroc, compute_eer
from ml.models.signature_evm import SignatureModel, evm_score, fit_evm, in_batch_negative_loss

ENROLLED = ["anjali", "barath"]
SIGNATURE_DIM = 32
D_MODEL = 32
EPOCHS = 300  # matches the reference project's published recipe / train_signature_evm_day3day4.py;
# that script found 40 epochs left the contrastive loss barely off its random-chance baseline.
STEPS_PER_EPOCH = 40
LR = 1e-3
SEED = 0
EVM_TAU = 20
EVM_CENTROID_ONLY = True  # see fit_evm's docstring: per-point EVM scored far below nearest-centroid
# on this project's small per-identity point counts; centroid_only fixes that directly.
ACCEPT_THRESHOLD = 0.18  # carried over from train_signature_evm_day3day4.py's centroid_only-mode
# calibration -- re-derive via the EER threshold below if this dataset's score distribution differs.

RAW_CONFIG = PreprocessConfig(name="raw_ch6_crossday")  # every optional stage off: no outlier
# removal, no smoothing, no normalization -- only the unavoidable link/PHY validation and time
# resampling that ablation_ch6_pipeline always applies.

CHECKPOINT_PATH = REPO_ROOT / "ml/checkpoints/signature_evm_ch6_crossday.pt"
LOG_PATH = REPO_ROOT / "ml/evaluation/results/signature_evm_ch6_crossday_log.csv"
LOG_FIELDNAMES = ["timestamp", "stage", "identity", "auroc", "eer", "genuine_accept_rate",
                   "correct_identity_rate", "false_accept_rate", "n_train", "n_test", "notes"]

TASK_NAME = "taskG_identity_auth_unauth"


def _task_identity_auth_unauth(window_index):
    """authorized + unauthorized windows, y = person_id (every identity, not just enrolled) --
    `none` (empty room) is excluded, this is a person-identification task, not presence."""
    mask = window_index["label"].isin(["authorized", "unauthorized"]).values
    y = window_index.loc[mask, "person_id"].values
    return mask, y


TASKS[TASK_NAME] = _task_identity_auth_unauth


def log_rows(rows: list[dict]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDNAMES})


def build_context_groups(dataset: CsiWindowDataset, groups: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per identity, sub-group its window positions by (date, motion) so the contrastive sampler can
    force query/gallery across different recording contexts (see pk_sample_batch) -- ported
    unchanged from train_signature_evm_day3day4.py."""
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
    aligned by identity so in_batch_negative_loss's diagonal is the correct-identity pairing. Query
    and gallery are drawn from DIFFERENT (date, motion) contexts whenever available, so the loss
    rewards identity signal that survives a context change."""
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{EPOCHS}: mean loss={total_loss / STEPS_PER_EPOCH:.4f}", flush=True)
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


def main(seed: int) -> None:
    print("building channel-6 manifest across 2026-09-15/16/17 (authorized + unauthorized only)...",
          flush=True)
    manifest = build_ch6_manifest_all_labels()
    manifest = manifest[manifest["label"] != "none"].reset_index(drop=True)
    print(f"  {len(manifest)} sessions: {manifest.groupby(['date', 'label']).size().to_dict()}", flush=True)

    target_rate_hz = compute_dataset_target_rate_hz(manifest, RAW_CONFIG)
    window_index = build_or_load_window_index(manifest, RAW_CONFIG, target_rate_hz)
    print(f"  window index: {len(window_index)} windows across dates "
          f"{sorted(window_index['date'].unique())}", flush=True)

    day_split = day_disjoint_split(window_index)
    assert day_split is not None, "need >1 date for a day-disjoint cross-day split"
    test_date, train_idx, test_idx = day_split
    assert_no_group_leakage(window_index, train_idx, test_idx, "session_dir")
    print(f"\nday-disjoint split: TEST DATE = {test_date} (held out entirely), "
          f"TRAIN = every other date. train_windows={len(train_idx)} test_windows={len(test_idx)}",
          flush=True)

    train_df = window_index.iloc[train_idx].reset_index(drop=True)
    test_df = window_index.iloc[test_idx].reset_index(drop=True)

    dataset_train = CsiWindowDataset(train_df, TASK_NAME)
    # classes derived independently (not shared with train) -- test-only identities (e.g. a
    # stranger never seen in training) must not KeyError against a train-only class list.
    dataset_test = CsiWindowDataset(test_df, TASK_NAME)
    n_subcarriers = dataset_train[0][0].shape[-1]
    print(f"train identities: {sorted(dataset_train.classes)}", flush=True)
    print(f"test identities:  {sorted(dataset_test.classes)} "
          f"(any not in train = genuinely novel strangers)", flush=True)
    novel = sorted(set(dataset_test.classes) - set(dataset_train.classes))
    if novel:
        print(f"  test identities NEVER seen in training: {novel}", flush=True)

    print("\n=== training signature encoder on TRAIN dates only "
          "(in-batch-negative contrastive loss) ===", flush=True)
    model = train_encoder(dataset_train, n_subcarriers, seed)

    print("\n=== embedding train + test windows with the frozen encoder ===", flush=True)
    train_embeddings = embed_all(model, dataset_train)
    test_embeddings = embed_all(model, dataset_test)

    print("\n=== fitting EVM on TRAIN identities only, scoring the held-out day ===", flush=True)
    per_identity_train = group_by_identity(train_embeddings, dataset_train.y, np.arange(len(dataset_train)))
    evm = fit_evm(per_identity_train, tau=EVM_TAU, equalize=True, centroid_only=EVM_CENTROID_ONLY)

    psi = evm_score(evm, test_embeddings, ENROLLED)
    y_true = np.isin(dataset_test.y, ENROLLED).astype(int)
    accept_score = psi.max(axis=1)
    pred_identity = np.array(ENROLLED)[psi.argmax(axis=1)]

    auroc = compute_auroc(y_true, accept_score)
    eer, eer_thr = compute_eer(y_true, accept_score)
    accept_pred = accept_score >= ACCEPT_THRESHOLD
    genuine_mask = y_true == 1
    genuine_accept_rate = float(accept_pred[genuine_mask].mean()) if genuine_mask.any() else float("nan")
    correct_identity_rate = float((pred_identity[genuine_mask] == dataset_test.y[genuine_mask]).mean()) \
        if genuine_mask.any() else float("nan")
    false_accept_rate = float(accept_pred[~genuine_mask].mean()) if (~genuine_mask).any() else float("nan")

    print(f"\n=== CROSS-DAY RESULT: train={sorted(set(window_index['date']) - {test_date})} "
          f"-> test={test_date} (n_test={len(test_idx)}) ===", flush=True)
    print(f"  AUROC={auroc:.3f}  EER={eer:.3f} (thr={eer_thr:.3f})  "
          f"genuine_accept_rate={genuine_accept_rate:.1%}  "
          f"correct_identity_rate={correct_identity_rate:.1%}  "
          f"false_accept_rate(any stranger)={false_accept_rate:.1%}  "
          f"(accept_threshold={ACCEPT_THRESHOLD})", flush=True)

    print(f"\n  per-enrolled-identity genuine accept rate on {test_date}:", flush=True)
    ts = datetime.now(timezone.utc).isoformat()
    log_rows_out = [{
        "timestamp": ts, "stage": "crossday_overall", "identity": "", "auroc": auroc, "eer": eer,
        "genuine_accept_rate": genuine_accept_rate, "correct_identity_rate": correct_identity_rate,
        "false_accept_rate": false_accept_rate, "n_train": len(train_idx), "n_test": len(test_idx),
        "notes": f"train={sorted(set(window_index['date']) - {test_date})} test={test_date}",
    }]
    for name in ENROLLED:
        m = genuine_mask & (dataset_test.y == name)
        if not m.any():
            continue
        rate = float(accept_pred[m].mean())
        correct = float((pred_identity[m] == name).mean())
        print(f"    {name:<10} n={int(m.sum()):>4}  accepted={rate:.1%}  correctly-identified={correct:.1%}",
              flush=True)
        log_rows_out.append({
            "timestamp": ts, "stage": "crossday_per_enrolled", "identity": name, "auroc": "", "eer": "",
            "genuine_accept_rate": rate, "correct_identity_rate": correct, "false_accept_rate": "",
            "n_train": "", "n_test": int(m.sum()), "notes": test_date,
        })

    print(f"\n  per-stranger false-accept rate on {test_date} (0% = correctly rejected every time):",
          flush=True)
    for name in sorted(set(dataset_test.y[~genuine_mask].tolist())):
        m = (~genuine_mask) & (dataset_test.y == name)
        fa = float(accept_pred[m].mean())
        flag = "  <-- NEVER SEEN IN TRAINING" if name in novel else ""
        print(f"    {name:<12} n={int(m.sum()):>4}  false_accept={fa:.1%}{flag}", flush=True)
        log_rows_out.append({
            "timestamp": ts, "stage": "crossday_per_stranger", "identity": name, "auroc": "", "eer": "",
            "genuine_accept_rate": "", "correct_identity_rate": "", "false_accept_rate": fa,
            "n_train": "", "n_test": int(m.sum()), "notes": "novel" if name in novel else "seen_in_train",
        })
    log_rows(log_rows_out)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "arch_kwargs": dict(n_subcarriers=n_subcarriers, d_model=D_MODEL, signature_dim=SIGNATURE_DIM),
        "evm": {name: evm[name] for name in ENROLLED},
        "enrolled": ENROLLED,
        "trained_identities": list(dataset_train.classes),
        "evm_tau": EVM_TAU,
        "accept_threshold": ACCEPT_THRESHOLD,
        "meta": {
            "train_dates": sorted(set(window_index["date"]) - {test_date}), "test_date": test_date,
            "target_channel": 6, "target_rate_hz": target_rate_hz,
            "crossday_auroc": auroc, "crossday_eer": eer,
            "crossday_genuine_accept_rate": genuine_accept_rate,
            "crossday_correct_identity_rate": correct_identity_rate,
            "crossday_false_accept_rate": false_accept_rate,
            "seed": seed, "created": datetime.now(timezone.utc).isoformat(),
            "notes": ("Single-ESP32 (no MultiBranchEncoder fusion needed) port of the WiFi-Sense "
                      "signature+EVM home model, evaluated on a genuinely unseen future day "
                      f"({test_date}), not a same-pool holdout. EVM fit only on train-date "
                      "identities; test-day-only strangers are the honest novel-impostor case."),
        },
    }, CHECKPOINT_PATH)
    print(f"\nsaved -> {CHECKPOINT_PATH}")
    print(f"log -> {LOG_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    main(seed=args.seed)
