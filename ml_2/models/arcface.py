"""ArcFace (additive angular margin) open-set head -- second embedding-based candidate alongside
prototypical.py, on the SAME backbone, so the two can be compared directly rather than confounded with
a different backbone too. At inference, the open-set decision uses max cosine similarity to any known
class's weight vector as the genuine score (no margin applied at eval time, matching standard ArcFace
practice: the margin only shapes training).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.01)
        self.scale = scale
        self.margin = margin

    def cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, dim=-1)
        return embeddings @ w.t()  # embeddings are already L2-normalized by the backbone

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = self.cosine(embeddings).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logit = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).float()
        logits = cosine * (1 - one_hot) + target_logit * one_hot
        return logits * self.scale

    @torch.no_grad()
    def genuine_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Max cosine similarity to any known identity's weight vector -- higher = more like SOME
        known identity, i.e. the open-set "authorized" score before any per-identity assignment."""
        return self.cosine(embeddings).max(dim=-1).values
