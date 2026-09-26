"""Signature embedding model (WhoFi-style) for open-set person identification, paired with an
Extreme Value Machine (EVM) for the actual accept/reject decision. Ported from the sibling
WiFi-Sense/csi_pipeline project's `models.py` (`SignatureModel`) and `person_openset_scores.py`
(`fit_evm_all`/`evm_score_all`, the published "tau20" control config), adapted for THIS project's
single-ESP32-node data -- that project's `MultiBranchEncoder`/`SharedNodeEncoder` exist specifically
to fuse MULTIPLE physical receivers, which doesn't apply here, so this uses one plain encoder branch.

Reuses `ml.models.common.BranchEncoder` (this project's own transformer-encoder block: linear input
projection + sinusoidal positional encoding + N standard transformer-encoder layers) instead of
re-porting a duplicate class -- it's architecturally identical to the reference project's
`TransformerEncoder`.

Why EVM instead of a closed-set classifier: `train_day3_day4_ch6_model.py`'s taskD checkpoint (a
plain softmax classifier) measured a 68.5% mean false-accept rate against a genuinely novel stranger
-- a closed-set classifier structurally cannot say "unknown," it only ranks the classes it was
trained on. The reference project measured the same failure mode with nearest-centroid rejection
(0% genuine accept at any threshold on its own held-out day) and found EVM's per-training-point
Weibull tail fit was the first thing that worked at all for this. This is a first pass at that idea
here, NOT the reference project's full production system -- see this module's and the training
script's docstrings for what's deliberately left out (cohort/T-norm normalization, per-identity EER
thresholds, multi-seed ensembling, walking-gating, temporal smoothing), all called out there as the
biggest remaining false-accept-rate levers, not yet implemented in this port.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import weibull_min

from ml.models.common import BranchEncoder

EPS = 1e-6


class SignatureHead(nn.Module):
    """Linear projection to the signature dimension, then L2-normalize onto the unit hypersphere."""

    def __init__(self, input_dim: int, signature_dim: int = 32):
        super().__init__()
        self.linear = nn.Linear(input_dim, signature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), p=2, dim=-1)


class SignatureModel(nn.Module):
    """Encoder + signature head. Amplitude-only, matching WhoFi's own design (see
    transformer_whofi.py's docstring for why phase is dropped in this model family) and matching
    the reference project's explicit finding that z-scoring against an empty-room baseline was
    neutral-to-harmful for this specific contrastive setup -- so, deliberately, NO calibration is
    applied before this model, unlike every other model in this project's zoo."""

    def __init__(self, n_subcarriers: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64,
                 dropout: float = 0.2, num_layers: int = 1, signature_dim: int = 32):
        super().__init__()
        self.encoder = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers)
        self.head = SignatureHead(d_model, signature_dim)

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.encoder(amplitude)      # (B, T, d_model)
        pooled = tokens.mean(dim=1)           # (B, d_model)
        return self.head(pooled)              # (B, signature_dim), L2-normalized


def in_batch_negative_loss(query_sig: torch.Tensor, gallery_sig: torch.Tensor) -> torch.Tensor:
    """DPR-style in-batch negatives (Karpukhin et al.), the WhoFi paper's actual loss: index i in
    query/gallery must be two DIFFERENT windows of the SAME identity -- that pairing is the training
    script's sampler's job, not this function's. sim(q,g) = Sq . Sg^T (dot product; both already
    L2-normalized, so this is cosine similarity), cross-entropy across each row so the diagonal
    (query i matched with gallery i, the same identity) is pushed toward 1 and every off-diagonal
    (a different identity) is pushed down."""
    sim = query_sig @ gallery_sig.T
    targets = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, targets)


MAX_SANE_SCALE = 8.0  # 4x the largest physically possible distance between two unit vectors (2.0)
MAX_SANE_SHAPE = 20.0  # a diagnostic (_diagnose_signature_evm.py) found shape blowing up to 82.4 on
# points whose tau nearest-other distances were tightly clustered (std as low as 0.008) -- a huge
# shape turns exp(-(d/scale)^shape) into a near step function (accept only almost exactly at that
# point's own distance, reject everything else), which doesn't generalize; well-behaved fits in the
# same run had shape in the single digits to ~25 at most.


def fit_evm(per_identity: dict[str, np.ndarray], tau: int = 20, equalize: bool = True,
            seed: int = 0, centroid_only: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-training-point Weibull tail fit, ported from person_openset_scores.py::fit_evm_all
    (its published 'tau20' control config: tau=20 nearest-OTHER-identity distances per point,
    equalize=True caps every identity to the smallest identity's point count so no identity's EVM
    is denser than another's purely from having more training windows -- otherwise the max-over-
    support-points score is mechanically higher for whichever identity had more data).

    `centroid_only=True` (added after a diagnostic, _diagnose_signature_evm.py, found this project's
    small per-identity point counts -- ~289 after equalize-capping -- make ~289 INDIVIDUAL per-point
    tail fits too noisy: a plain nearest-centroid rule on the same embeddings scored mean AUROC 0.737
    vs. the per-point EVM's 0.408-0.466 even after bounding degenerate shape/scale values): collapses
    "own" to a single point (that identity's centroid) before fitting, so exactly ONE stable Weibull
    tail is fit per identity instead of hundreds of noisy ones -- same statistical-distance idea,
    much lower variance. `equalize` is skipped in this mode (every identity trivially contributes
    exactly 1 point already).

    Returns {identity: (own_points, weibull_params)}; weibull_params[i] = (shape, scale) fitted to
    point i's own `tau` nearest OTHER-identity distances via scipy's weibull_min.fit(floc=0)."""
    rng = np.random.RandomState(seed)
    names = sorted(per_identity)
    n_cap = min(len(per_identity[n]) for n in names) if (equalize and not centroid_only) else None
    evm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        own = per_identity[name].mean(axis=0, keepdims=True) if centroid_only else per_identity[name]
        if equalize and not centroid_only and len(own) > n_cap:
            own = own[rng.choice(len(own), size=n_cap, replace=False)]
        others = np.concatenate([p for n_, p in per_identity.items() if n_ != name], axis=0)
        # MAX_SANE_SCALE: embeddings are L2-normalized, so euclidean distance between any two of
        # them is bounded in [0, 2] -- a fitted scale far outside that range (seen in practice: 98.4,
        # from a point whose 20 nearest-other distances were all crammed into a near-zero-variance
        # band, e.g. std=0.016) means weibull_min.fit degenerated, not that the tail is genuinely
        # that wide. evm_score's max-over-points aggregation has NO robustness to even one such
        # point (see module docstring) -- to keep this diagnostic-driven fix simple, catch it here
        # at fit time instead: fall back to the safe (near.mean()) estimate whenever the fit lands
        # outside the physically possible range.
        params = np.empty((len(own), 2), dtype=np.float64)
        for i in range(0, len(own), 256):
            blk = own[i:i + 256]
            d = np.linalg.norm(blk[:, None, :] - others[None, :, :], axis=2)
            k = min(tau, d.shape[1])
            near = np.sort(d, axis=1)[:, :k]
            for r in range(len(blk)):
                try:
                    shape, _, scale = weibull_min.fit(near[r], floc=0)
                    if (not (np.isfinite(shape) and np.isfinite(scale))
                            or scale > MAX_SANE_SCALE or shape > MAX_SANE_SHAPE):
                        raise ValueError("degenerate fit")
                except Exception:
                    shape, scale = 1.0, float(near[r].mean() + EPS)
                params[i + r] = (shape, scale)
        evm[name] = (own, params)
    return evm


def evm_score(evm: dict[str, tuple[np.ndarray, np.ndarray]], emb: np.ndarray, names: list[str],
              chunk: int = 256, top_k: int = 5) -> np.ndarray:
    """Probability-of-sample-inclusion (psi) for every embedding, against every named identity's
    fitted EVM -- ported from person_openset_scores.py::evm_score_all, with one deliberate change:
    the ORIGINAL takes max() over an identity's stored support points, which a diagnostic
    (ml/training/_diagnose_signature_evm.py) found has zero robustness to even a single degenerate
    per-point fit -- one bad point can single-handedly make an identity's EVM accept almost
    anything, since max() only needs ONE point to agree. This instead averages the `top_k` highest
    psi values per identity (mean of the best-agreeing points, not just the single best), which
    still rewards a genuine close match but can't be dominated by one outlier."""
    out = np.zeros((len(emb), len(names)), dtype=np.float64)
    for ci, name in enumerate(names):
        pts, params = evm[name]
        shape, scale = params[:, 0], params[:, 1]
        k = min(top_k, len(pts))
        for i in range(0, len(emb), chunk):
            d = np.linalg.norm(emb[i:i + chunk, None, :] - pts[None, :, :], axis=2)
            psi = np.exp(-(d / (scale[None, :] + EPS)) ** shape[None, :])
            out[i:i + chunk, ci] = np.sort(psi, axis=1)[:, -k:].mean(axis=1)
    return out
