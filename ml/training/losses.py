"""Contrastive loss for the WhoFi-style embedding model (Task C, verification) + EER threshold search.
In-batch-negative / supervised-contrastive style, matching WhoFi's description: every other sample in
the batch with a different label is a negative, no separate triplet mining needed.
"""
from __future__ import annotations

import torch


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
