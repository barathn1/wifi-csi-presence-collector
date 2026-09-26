"""Best-effort reconstruction of Wi-Gait's classifier -- "stacked multi-scale residual CNN enhanced
Transformer encoder" per its abstract/citations -- NOT verified against the paper (paywalled, no
accessible preprint/source; see data_pipeline/wigait_approx_preprocessing.py for the same caveat on
the preprocessing side). One multi-scale residual CNN block (kernel sizes 3/5/7) feeding one
transformer encoder layer per branch (torso motion, limb-swing motion), late-fused like this repo's
other dual-branch models -- "stacked" is simplified to a single CNN block for tractability.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.common import SinusoidalPositionalEncoding


class MultiScaleResidualCNN(nn.Module):
    def __init__(self, n_subcarriers: int, d_model: int, kernel_sizes: tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(n_subcarriers, d_model, k, padding=k // 2) for k in kernel_sizes
        ])
        self.merge = nn.Conv1d(d_model * len(kernel_sizes), d_model, 1)
        self.residual = nn.Conv1d(n_subcarriers, d_model, 1)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, n_sub, T)
        feats = torch.cat([self.act(b(x)) for b in self.branches], dim=1)
        merged = self.merge(feats)
        return self.norm(merged + self.residual(x))


class WiGaitBranch(nn.Module):
    def __init__(self, n_subcarriers: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64, dropout: float = 0.2):
        super().__init__()
        self.cnn = MultiScaleResidualCNN(n_subcarriers, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, n_sub, T) -> (B, T, d_model)
        h = self.cnn(x).transpose(1, 2)
        h = self.pos_encoding(h)
        return self.encoder(h)


class WiGaitApprox(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32, n_heads: int = 4,
                 d_ff: int = 64, dropout: float = 0.2):
        super().__init__()
        self.torso_branch = WiGaitBranch(n_subcarriers, d_model, n_heads, d_ff, dropout)
        self.limb_branch = WiGaitBranch(n_subcarriers, d_model, n_heads, d_ff, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes),
        )

    def forward(self, torso: torch.Tensor, limb: torch.Tensor) -> torch.Tensor:
        t = self.torso_branch(torso).mean(dim=1)
        l = self.limb_branch(limb).mean(dim=1)
        return self.classifier(torch.cat([t, l], dim=-1))
