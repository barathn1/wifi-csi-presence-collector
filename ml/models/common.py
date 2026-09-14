"""Shared building blocks for the transformer model zoo -- kept deliberately small (d_model=32, 1
layer) per RESEARCH_NOTES.md's finding (both WhoFi and the ESP32 person-ID paper) that going deeper
hurt on this class of small CSI dataset, not helped.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
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
    """Input projection + positional encoding + N-layer transformer encoder (Attention Is All You Need:
    multi-head self-attention -> Add&Norm -> position-wise feed-forward -> Add&Norm, per layer), for
    one channel (amplitude or phase). Defaults (d_model=32, 4 heads, d_ff=64, 1 layer, post-LN) mirror
    the ESP32 person-ID paper's per-branch design; `num_layers`/`norm_first` are exposed so the depth
    and pre-LN-vs-post-LN axes can be swept (see run_day2_sweep.py's Stage T1) rather than assumed."""

    def __init__(self, n_subcarriers: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64, dropout: float = 0.2,
                 num_layers: int = 1, norm_first: bool = False):
        super().__init__()
        self.input_proj = nn.Linear(n_subcarriers, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, n_subcarriers) -> (B, T, d_model)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        return self.encoder(x)


class CrossAttentionBlock(nn.Module):
    """One complete Transformer block applied to cross-attention instead of self-attention: multi-head
    attention (query from `x`, key/value from `context`) -> Add&Norm -> position-wise feed-forward ->
    Add&Norm -- the same two-sublayer structure "Attention Is All You Need" uses throughout, just with
    the attention sub-layer's K/V sourced from another stream. (The original cross-attention models in
    this repo only had the attention+norm sublayer, missing the FFN+norm sublayer entirely -- fixed
    here.) `norm_first` switches Add&Norm's ordering: post-LN (norm after the residual, the paper's
    original) vs pre-LN (norm before each sub-layer, more common in modern transformers)."""

    def __init__(self, d_model: int = 32, n_heads: int = 4, d_ff: int = 64, dropout: float = 0.2,
                 norm_first: bool = False):
        super().__init__()
        self.norm_first = norm_first
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
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
