"""Explicit multi-receiver fusion model -- the alternative to the default "each ESP is its own
independent sample" treatment used everywhere else in ml_2. Only meaningful for sessions that actually
have >1 receiver (2026-09-21/22 in this dataset). Weight-shared branch encoder across receivers (fewer
params than one branch per receiver, appropriate given only 2 days' worth of multi-receiver data);
pooled, L2-normalized per-receiver embeddings are concatenated (not averaged) before the classifier, so
it can still learn to weight one receiver over another.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_2.models.backbone import BranchEncoder


class ReceiverFusionTransformer(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, n_receivers: int = 3, d_model: int = 32,
                 n_heads: int = 4, d_ff: int = 64, dropout: float = 0.2, num_layers: int = 1,
                 norm_first: bool = False):
        super().__init__()
        self.n_receivers = n_receivers
        self.shared_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.classifier = nn.Sequential(nn.Linear(n_receivers * d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(d_model, n_classes))

    def embed_one(self, amplitude: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.shared_branch(amplitude).mean(dim=1), p=2, dim=-1)

    def forward(self, amplitudes: list[torch.Tensor]) -> torch.Tensor:
        assert len(amplitudes) == self.n_receivers, (len(amplitudes), self.n_receivers)
        fused = torch.cat([self.embed_one(a) for a in amplitudes], dim=-1)
        return self.classifier(fused)
