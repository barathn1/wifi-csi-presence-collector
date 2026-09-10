"""1D-CNN dual-branch baseline -- the ESP32 paper's "CNN-only" ablation point (98.72%, vs 99.82% for
their transformer) is the precedent for including this in the zoo despite the project's transformer
focus: it's the cheapest deep-learning floor to beat.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1DDualBranch(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, hidden_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        def branch():
            return nn.Sequential(
                nn.Conv1d(n_subcarriers, hidden_channels, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2), nn.ReLU(),
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
