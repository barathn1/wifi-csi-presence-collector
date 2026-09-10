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
    """Input projection + positional encoding + 1-layer transformer encoder, for one channel
    (amplitude or phase). Mirrors the ESP32 person-ID paper's per-branch design (d_model=32, 4 heads,
    d_ff=64, dropout=0.2)."""

    def __init__(self, n_subcarriers: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_subcarriers, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, n_subcarriers) -> (B, T, d_model)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        return self.encoder(x)
