"""Second-stage decision over a SEQUENCE of per-window embeddings, for "watch someone for 10-30s
before deciding" instead of deciding from one ~1s window. `evaluation/segment_aggregation.py` already
does this by plain-averaging per-window scores (a free, zero-parameter win per ARGUS). This is the
learned alternative: attention-pool the sequence of embeddings itself, one level up from the per-window
models' own internal attention over packets -- same "Attention Is All You Need" mechanism (multi-head
attention, here with a single learned query instead of a token-derived one, i.e. the same shape as a
CLS-token pooling head), applied at the segment level instead of the packet level.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPoolClassifier(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, n_classes),
        )

    def forward(self, window_embeddings: torch.Tensor) -> torch.Tensor:  # (B, N, embed_dim) -> (B, n_classes)
        query = self.query.expand(window_embeddings.shape[0], -1, -1)
        pooled, _ = self.attn(query, window_embeddings, window_embeddings)
        pooled = self.norm(pooled.squeeze(1))
        return self.classifier(pooled)
