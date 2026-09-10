"""Replica of the ESP32 person-ID paper's architecture (arXiv:2507.12854, 99.82% on 6 subjects):
amplitude and phase encoded by separate 1-layer transformer branches, mean-pooled, concatenated, and
passed through a linear head -- late fusion, no cross-branch interaction. This is the baseline that
transformer_crossattn.py (the "own idea" model) is compared against.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.common import BranchEncoder


class DualBranchTransformer(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.amp_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout)
        self.phase_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes),
        )

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_tokens = self.amp_branch(amplitude)      # (B, T, d_model)
        phase_tokens = self.phase_branch(phase)      # (B, T, d_model)
        fused = torch.cat([amp_tokens.mean(dim=1), phase_tokens.mean(dim=1)], dim=-1)
        return self.classifier(fused)

    def embed(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Pooled fused representation, pre-classifier -- for embedding/verification-style evaluation."""
        amp_tokens = self.amp_branch(amplitude)
        phase_tokens = self.phase_branch(phase)
        return torch.cat([amp_tokens.mean(dim=1), phase_tokens.mean(dim=1)], dim=-1)
