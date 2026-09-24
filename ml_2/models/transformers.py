"""Classifier-head transformer variants (softmax, not distance/margin-based) -- self-contained
re-implementations of the WhoFi / dual-branch / cross-attention family, kept for direct comparison
against the open-set embedding models (prototypical.py, arcface.py)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_2.models.backbone import BranchEncoder, CrossAttentionBlock


class WhoFiTransformer(nn.Module):
    """Single amplitude-only branch, L2-normalized embedding -> linear classifier."""

    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32, n_heads: int = 4,
                 d_ff: int = 64, dropout: float = 0.2, num_layers: int = 1, norm_first: bool = False):
        super().__init__()
        self.branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        del phase
        pooled = F.normalize(self.branch(amplitude).mean(dim=1), p=2, dim=-1)
        return self.classifier(pooled)


class DualBranchTransformer(nn.Module):
    """Separate amplitude/phase branches, late-fusion concat -> classifier."""

    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32, n_heads: int = 4,
                 d_ff: int = 64, dropout: float = 0.2, num_layers: int = 1, norm_first: bool = False):
        super().__init__()
        self.amp_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.phase_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.classifier = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(d_model, n_classes))

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp = self.amp_branch(amplitude).mean(dim=1)
        pha = self.phase_branch(phase).mean(dim=1)
        return self.classifier(torch.cat([amp, pha], dim=-1))


class CrossAttentionTransformer(nn.Module):
    """Amplitude and phase branches bidirectionally cross-attend before pooling+classifying."""

    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32, n_heads: int = 4,
                 d_ff: int = 64, dropout: float = 0.2, num_layers: int = 1, norm_first: bool = False):
        super().__init__()
        self.amp_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.phase_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.amp_cross = CrossAttentionBlock(d_model, n_heads, d_ff, dropout, norm_first)
        self.phase_cross = CrossAttentionBlock(d_model, n_heads, d_ff, dropout, norm_first)
        self.classifier = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(d_model, n_classes))

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_tok = self.amp_branch(amplitude)
        pha_tok = self.phase_branch(phase)
        amp_cross = self.amp_cross(amp_tok, pha_tok).mean(dim=1)
        pha_cross = self.phase_cross(pha_tok, amp_tok).mean(dim=1)
        return self.classifier(torch.cat([amp_cross, pha_cross], dim=-1))
