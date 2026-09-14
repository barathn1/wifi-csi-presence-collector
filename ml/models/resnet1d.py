"""1D-ResNet dual-branch -- the actual "ResNet" requested (cnn1d.py's CNN1DDualBranch has no skip
connections, it's a plain 2-layer CNN despite living next to the transformer zoo). Same dual-branch/
concat/classifier shape as CNN1DDualBranch so the only difference under test is the residual connection
itself: each block is Conv1d-BN-ReLU-Conv1d-BN, `+` identity skip, then ReLU -- the standard ResNet
basic block, adapted to 1D (time axis) with subcarriers as channels.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)


class ResNet1DDualBranch(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, hidden_channels: int = 32, n_blocks: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        def branch():
            return nn.Sequential(
                nn.Conv1d(n_subcarriers, hidden_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_channels), nn.ReLU(),
                *[ResidualBlock1D(hidden_channels, dropout) for _ in range(n_blocks)],
                nn.AdaptiveAvgPool1d(1),
            )
        self.amp_branch = branch()
        self.phase_branch = branch()
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, n_classes),
        )

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_feat = self.amp_branch(amplitude.transpose(1, 2)).squeeze(-1)   # (B, T, n_sub) -> (B, hidden)
        phase_feat = self.phase_branch(phase.transpose(1, 2)).squeeze(-1)
        return self.classifier(torch.cat([amp_feat, phase_feat], dim=-1))
