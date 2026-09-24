"""Self-contained transformer building blocks (no ml.models import) -- kept small (d_model=32, 1
layer) since this class of small CSI dataset tends to overfit deeper models.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class BranchEncoder(nn.Module):
    """Input projection + positional encoding + N-layer transformer encoder, for one channel
    (amplitude or phase)."""

    def __init__(self, n_subcarriers: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64,
                 dropout: float = 0.2, num_layers: int = 1, norm_first: bool = False):
        super().__init__()
        self.input_proj = nn.Linear(n_subcarriers, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, n_subcarriers) -> (B, T, d_model)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        return self.encoder(x)


class CrossAttentionBlock(nn.Module):
    """One transformer block applied to cross-attention: multi-head attention (query from `x`, key/
    value from `context`) -> Add&Norm -> feed-forward -> Add&Norm."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, d_ff: int = 64, dropout: float = 0.2,
                 norm_first: bool = False):
        super().__init__()
        self.norm_first = norm_first
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        out, _ = self.cross_attn(query=query, key=context, value=context)
        return out

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = x + self.dropout(self._attend(self.norm1(x), context))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout(self._attend(x, context)))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DualBranchEmbedding(nn.Module):
    """Amplitude + phase branches, mean-pooled + projected to an L2-normalized embedding -- the shared
    backbone for the open-set embedding models (prototypical.py, arcface.py)."""

    def __init__(self, n_subcarriers: int, embed_dim: int = 32, d_model: int = 32, n_heads: int = 4,
                 d_ff: int = 64, dropout: float = 0.2, num_layers: int = 1, norm_first: bool = False):
        super().__init__()
        self.amp_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.phase_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.project = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                       nn.Linear(d_model, embed_dim))

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_tokens = self.amp_branch(amplitude).mean(dim=1)
        phase_tokens = self.phase_branch(phase).mean(dim=1)
        fused = torch.cat([amp_tokens, phase_tokens], dim=-1)
        return F.normalize(self.project(fused), p=2, dim=-1)
