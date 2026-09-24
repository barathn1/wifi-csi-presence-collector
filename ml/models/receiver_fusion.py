"""Does reading all N ESP32 receivers for a window (instead of just one) help authorized/non-authorized
detection? Only 2026-09-21/22 ever had more than one receiver recording in parallel (3 boards), so this
model -- and the comparison script that trains it, ml/training/compare_receiver_fusion.py -- is scoped
to just those 2 dates.

Architecture: each receiver's amplitude stream goes through the SAME (weight-shared) BranchEncoder --
sharing weights instead of giving each receiver its own branch, because this comparison's training set
is only 2 days' worth of data (far smaller than the 5-day pool used elsewhere in this project), and a
shared encoder has 1/3 the parameters of 3 independent ones for the same architecture. The 3 receivers'
pooled, L2-normalized embeddings are concatenated (not averaged) before the classifier, so the
classifier can still learn to weight one receiver over another rather than being forced to treat them
identically -- the receive-diversity idea from RESEARCH_NOTES.md section 2 (single-antenna spatial-
diversity limits being the likely reason ESP32 struggles at simultaneous multi-person ID), just applied
to multiple independent single-antenna receivers instead of multiple antennas on one receiver.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.common import BranchEncoder


class ReceiverFusionTransformer(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, n_receivers: int = 3, d_model: int = 32,
                 n_heads: int = 4, d_ff: int = 64, dropout: float = 0.2, num_layers: int = 1,
                 norm_first: bool = False):
        super().__init__()
        self.n_receivers = n_receivers
        self.shared_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.classifier = nn.Sequential(
            nn.Linear(n_receivers * d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def embed_one(self, amplitude: torch.Tensor) -> torch.Tensor:
        tokens = self.shared_branch(amplitude)
        return F.normalize(tokens.mean(dim=1), p=2, dim=-1)

    def forward(self, amplitudes: list[torch.Tensor]) -> torch.Tensor:
        """`amplitudes`: list of `n_receivers` tensors, each (B, T, n_subcarriers), one per receiver,
        same window/time-slice, already time-normalized onto a shared rate so they're comparable."""
        assert len(amplitudes) == self.n_receivers, (len(amplitudes), self.n_receivers)
        fused = torch.cat([self.embed_one(a) for a in amplitudes], dim=-1)
        return self.classifier(fused)
