"""BiLSTM dual-branch encoder -- WhoFi's own fallback option when the Transformer proved unstable
(it wasn't, on their data, but they tested it; kept here for completeness of the model zoo)."""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMDualBranch(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, hidden_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.amp_lstm = nn.LSTM(n_subcarriers, hidden_size, batch_first=True, bidirectional=True)
        self.phase_lstm = nn.LSTM(n_subcarriers, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, n_classes),
        )

    def _pool(self, lstm: nn.LSTM, x: torch.Tensor) -> torch.Tensor:
        out, _ = lstm(x)          # (B, T, 2*hidden)
        return out.mean(dim=1)    # mean-pool over time, matches the transformer branches' pooling

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self._pool(self.amp_lstm, amplitude), self._pool(self.phase_lstm, phase)], dim=-1)
        return self.classifier(fused)
