"""Diagnose why train_signature_evm_day3day4.py's checkpoint scored AUROC < 0.5 (worse than random)
on the leave-one-stranger-out eval, with at least one fold (sumanth) accepting 100% of everything
regardless of true identity. Two live hypotheses, tested directly rather than guessed:

1. The contrastive encoder never actually learned to separate identities (embeddings collapsed) --
   tested via intra-identity vs inter-identity cosine-distance distributions on the unit hypersphere.
2. The per-point Weibull tail fits are degenerating (near-flat decay -> psi stays near 1 for any
   distance) -- tested by inspecting the actual (shape, scale) parameter distributions and the raw
   tau-nearest-other distances a few fits were computed from.

    python3 -m ml.training._diagnose_signature_evm
"""
from __future__ import annotations

import numpy as np
import torch

from ml.data_pipeline.decode_csi import REPO_ROOT
from ml.data_pipeline.splits import leave_one_unauthorized_person_out
from ml.data_pipeline.torch_dataset import CsiWindowDataset
from ml.models.signature_evm import SignatureModel, fit_evm
from ml.training.train_signature_evm_day3day4 import (
    build_or_load_window_index, embed_all, group_by_identity, EVM_TAU,
)
from ml.training.train_day3_day4_ch6_model import build_day3_day4_ch6_manifest, compute_dataset_target_rate_hz

CHECKPOINT_PATH = REPO_ROOT / "ml/checkpoints/signature_evm_day3day4.pt"


def main() -> None:
    print("loading checkpoint...", flush=True)
    ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = SignatureModel(**ck["arch_kwargs"])
    model.load_state_dict(ck["model_state"])
    model.eval()

    print("rebuilding window index (should hit the existing cache)...", flush=True)
    manifest = build_day3_day4_ch6_manifest()
    target_rate_hz = compute_dataset_target_rate_hz(manifest)
    window_index = build_or_load_window_index(manifest, target_rate_hz)
    dataset = CsiWindowDataset(window_index, "taskF_person_identity_all", calibration=None)
    print(f"  {len(dataset)} windows, identities: {dataset.classes}", flush=True)

    print("\nembedding every window...", flush=True)
    embeddings = embed_all(model, dataset)
    print(f"  embeddings shape={embeddings.shape}, mean norm={np.linalg.norm(embeddings, axis=1).mean():.4f} "
          f"(should be ~1.0, L2-normalized)", flush=True)

    # --- Hypothesis 1: did the embedding space actually separate identities? ---
    print("\n=== intra- vs inter-identity distance check (embeddings are L2-normalized, so "
          "euclidean distance in [0,2]; if the encoder learned nothing, these two numbers will be "
          "nearly identical) ===", flush=True)
    rng = np.random.default_rng(0)
    per_identity = group_by_identity(embeddings, dataset.y, np.arange(len(dataset)))
    names = sorted(per_identity)

    intra_dists = []
    for name in names:
        pts = per_identity[name]
        if len(pts) < 2:
            continue
        i = rng.integers(0, len(pts), size=min(2000, len(pts) * 5))
        j = rng.integers(0, len(pts), size=len(i))
        keep = i != j
        d = np.linalg.norm(pts[i[keep]] - pts[j[keep]], axis=1)
        intra_dists.append(d)
    intra_dists = np.concatenate(intra_dists)

    inter_dists = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            pa, pb = per_identity[names[a]], per_identity[names[b]]
            i = rng.integers(0, len(pa), size=min(500, len(pa)))
            j = rng.integers(0, len(pb), size=min(500, len(pb)))
            n = min(len(i), len(j))
            inter_dists.append(np.linalg.norm(pa[i[:n]] - pb[j[:n]], axis=1))
    inter_dists = np.concatenate(inter_dists)

    print(f"  intra-identity distance: mean={intra_dists.mean():.4f} std={intra_dists.std():.4f} "
          f"(n={len(intra_dists)})", flush=True)
    print(f"  inter-identity distance: mean={inter_dists.mean():.4f} std={inter_dists.std():.4f} "
          f"(n={len(inter_dists)})", flush=True)
    separation = (inter_dists.mean() - intra_dists.mean()) / (intra_dists.std() + inter_dists.std() + 1e-9)
    print(f"  separation score (higher = better; ~0 means no separation at all): {separation:.4f}", flush=True)

    print("\n  per-identity mean distance to EVERY other identity (should differ noticeably from "
          "identity to identity if geometry is meaningful; near-uniform values mean the encoder "
          "collapsed everyone to roughly the same region of the hypersphere):", flush=True)
    centroids = {n: per_identity[n].mean(axis=0) for n in names}
    for a in names:
        row = " ".join(f"{b}={np.linalg.norm(centroids[a]-centroids[b]):.3f}" for b in names if b != a)
        print(f"    {a:<12} {row}", flush=True)

    # --- Is the anjali/divya/sumanth clustering actually a DAY confound? ---
    print("\n=== day-confound check: anjali/barath split by recording date, compared to divya/sumanth "
          "(both of whom have sessions on BOTH days) ===", flush=True)
    dates = dataset.index["date"].values
    for enrolled in ("anjali", "barath"):
        mask_e = dataset.y == enrolled
        for date in sorted(set(dates[mask_e])):
            sub = embeddings[mask_e & (dates == date)]
            centroid = sub.mean(axis=0)
            row = " ".join(f"{other}={np.linalg.norm(centroid - centroids[other]):.3f}"
                           for other in ("anjali", "barath", "divya", "sumanth") if other != enrolled)
            print(f"    {enrolled} on {date} (n={len(sub)}): {row}", flush=True)

    # --- Isolate: is it the EMBEDDING that lacks separation, or just the EVM decision layer? ---
    # Simplest possible open-set rule on the SAME embeddings: nearest-centroid. If this ALSO scores
    # AUROC<0.5, the embedding itself carries no genuine-vs-impostor signal regardless of decision
    # layer; if this scores meaningfully >0.5, the EVM machinery specifically is the problem.
    print("\n=== nearest-centroid sanity check (same embeddings, simplest possible decision rule) ===",
          flush=True)
    from ml.evaluation.metrics import compute_auroc
    ENROLLED = ["anjali", "barath"]
    aurocs = []
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(dataset.index, include_none=False):
        per_identity_train = group_by_identity(embeddings, dataset.y, train_idx)
        enrolled_centroids = {n: per_identity_train[n].mean(axis=0) for n in ENROLLED}
        y_true = np.isin(dataset.y[test_idx], ENROLLED).astype(int)
        # accept_score = negative distance to nearest enrolled centroid (higher = closer = more genuine)
        dists = np.stack([np.linalg.norm(embeddings[test_idx] - enrolled_centroids[n], axis=1)
                           for n in ENROLLED], axis=1)
        accept_score = -dists.min(axis=1)
        auroc = compute_auroc(y_true, accept_score)
        aurocs.append(auroc)
        print(f"  held out '{held_out}': nearest-centroid auroc={auroc:.3f}", flush=True)
    print(f"  MEAN nearest-centroid auroc across {len(aurocs)} folds: {np.nanmean(aurocs):.3f}", flush=True)

    # --- Hypothesis 2: are the Weibull tail fits degenerating? ---
    print("\n=== Weibull fit diagnostics on the 'sumanth'-held-out fold (the degenerate one: "
          "genuine_accept=1.0 AND false_accept=1.0 simultaneously) ===", flush=True)
    for held_out, train_idx, test_idx in leave_one_unauthorized_person_out(dataset.index, include_none=False):
        if held_out != "sumanth":
            continue
        per_identity_train = group_by_identity(embeddings, dataset.y, train_idx)
        evm = fit_evm(per_identity_train, tau=EVM_TAU, equalize=True)
        for name in ["anjali", "barath"]:
            pts, params = evm[name]
            shape, scale = params[:, 0], params[:, 1]
            print(f"  {name}: n_points={len(pts)}", flush=True)
            print(f"    shape:  min={shape.min():.4f} median={np.median(shape):.4f} max={shape.max():.4f}",
                  flush=True)
            print(f"    scale:  min={scale.min():.4f} median={np.median(scale):.4f} max={scale.max():.4f}",
                  flush=True)
            # sample one point's actual tau-nearest-other distances that fed its Weibull fit
            own = pts
            others = np.concatenate([p for n_, p in per_identity_train.items() if n_ != name], axis=0)
            d0 = np.linalg.norm(own[0:1] - others, axis=1)
            near0 = np.sort(d0)[:EVM_TAU]
            print(f"    point 0's {EVM_TAU} nearest-other distances: min={near0.min():.4f} "
                  f"max={near0.max():.4f} std={near0.std():.6f} (near-zero std -> degenerate Weibull fit)",
                  flush=True)
        break


if __name__ == "__main__":
    main()
