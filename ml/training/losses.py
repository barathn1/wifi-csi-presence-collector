"""Contrastive loss for the WhoFi-style embedding model (Task C, verification) + EER threshold search.
In-batch-negative / supervised-contrastive style, matching WhoFi's description: every other sample in
the batch with a different label is a negative, no separate triplet mining needed.
"""
from __future__ import annotations

import torch


def triplet_semihard_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """PyTorch reimplementation of the semi-hard triplet mining idea behind
    `tfa.losses.TripletSemiHardLoss` (used by the walking-only BiLSTM in
    `ml/training/run_walk_bilstm_loo_day.py`) -- not a byte-identical port, same mining rule: for every
    (anchor, positive) pair, pick the closest negative that's still farther than the positive
    (semi-hard); if no such negative exists in this batch, fall back to the hardest (closest) negative
    overall rather than skipping the pair. embeddings: (B, D), L2-normalized INSIDE this function.
    labels: (B,) int identity labels.

    The spec's embedding layer has a linear activation with no normalization layer, but feeding raw,
    unnormalized embeddings into a margin-based pairwise-distance loss lets embedding norm grow
    unboundedly during training -- confirmed empirically on this dataset: with augmentation off, adding
    this loss on top of cross-entropy made train accuracy collapse to the majority-class baseline
    (~54%) and stay there for 15+ epochs, while cross-entropy alone reaches ~88%. Normalizing onto the
    unit hypersphere first (standard practice for triplet losses generally, e.g. FaceNet) bounds
    distances to [0, 2] so the margin=1.0 default is meaningful and the gradient scale can't run away.

    Fully vectorized over (anchor, positive, negative) triples via one (B, B, B) broadcast rather than
    a per-anchor Python loop -- fine at this project's batch sizes (<=128)."""
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)
    b = embeddings.size(0)
    labels = labels.view(-1, 1)
    same = labels == labels.T
    eye = torch.eye(b, dtype=torch.bool, device=embeddings.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    d_ap = dist.unsqueeze(2)                    # (B, B, 1): d_ap[i, j, :] = dist[i, j]
    d_ik = dist.unsqueeze(1).expand(b, b, b)     # (B, B, B): d_ik[i, :, k] = dist[i, k]
    neg_bcast = neg_mask.unsqueeze(1).expand(b, b, b)

    semihard = (d_ik > d_ap) & neg_bcast
    d_an_semihard = d_ik.masked_fill(~semihard, float("inf")).amin(dim=2)
    d_an_hardest = d_ik.masked_fill(~neg_bcast, float("-inf")).amax(dim=2)
    has_semihard = torch.isfinite(d_an_semihard)
    d_an = torch.where(has_semihard, d_an_semihard, d_an_hardest)

    loss_matrix = torch.clamp(dist - d_an + margin, min=0.0)
    if not pos_mask.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return loss_matrix[pos_mask].mean()


def in_batch_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """embeddings: (B, D), L2-normalized. labels: (B,) int identity labels."""
    device = embeddings.device
    sim = embeddings @ embeddings.T / temperature  # (B, B)
    n = sim.size(0)
    self_mask = torch.eye(n, dtype=torch.bool, device=device)

    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T) & ~self_mask

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = positive_mask.float().sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / pos_counts
    valid = positive_mask.any(dim=1)  # samples with zero positives in this batch contribute nothing
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return -mean_log_prob_pos[valid].mean()
