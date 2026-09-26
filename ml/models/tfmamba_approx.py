"""Best-effort reconstruction of TF-Mamba (IEEE, doi:10.1109/... 10817504) -- NOT a verified
replication; the IEEE Xplore page returned no extractable content and no preprint was found. Built
from the architectural concept confirmed via citations: a dual-stream model with a Time-Mamba block
(raw time-domain CSI) and a Frequency-Mamba block (2D-wavelet-transformed CSI), each a state-space
(Mamba) encoder, fused for classification.

Simplifications from the (unknown) original, stated plainly:
- The 2D DWT is a single-level Haar transform, LL (approximation) band only -- not necessarily the
  exact wavelet/level the paper uses.
- One Mamba block per stream (not necessarily their depth).
- TF-Mamba is an activity-recognition paper; here its architecture is repurposed for person
  identification (classifier head swapped to identity classes) to compare against this repo's
  existing WhoFi replica on the same task.
- Amplitude only (phase argument accepted for drop-in compatibility with this repo's existing
  training pipeline, but unused) -- the extracted description doesn't specify amplitude vs phase.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.mamba_block import MinimalMambaBlock


def haar_dwt2_ll(x: torch.Tensor) -> torch.Tensor:
    """Single-level 2D Haar DWT, LL (approximation) band only. x: (B, T, S) -> (B, T//2, S//2)."""
    B, T, S = x.shape
    if T % 2:
        x = x[:, :-1, :]
    if S % 2:
        x = x[:, :, :-1]
    a, b = x[:, 0::2, 0::2], x[:, 0::2, 1::2]
    c, d = x[:, 1::2, 0::2], x[:, 1::2, 1::2]
    return (a + b + c + d) / 2.0


class TFMambaApprox(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32,
                 d_state: int = 16, dropout: float = 0.2):
        super().__init__()
        n_freq_sub = max(1, n_subcarriers // 2)

        self.time_input_proj = nn.Linear(n_subcarriers, d_model)
        self.time_mamba = MinimalMambaBlock(d_model, d_state=d_state)

        self.freq_input_proj = nn.Linear(n_freq_sub, d_model)
        self.freq_mamba = MinimalMambaBlock(d_model, d_state=d_state)

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes),
        )

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        time_tokens = self.time_input_proj(amplitude)
        time_out = self.time_mamba(time_tokens) + time_tokens
        time_pooled = time_out.mean(dim=1)

        freq_input = haar_dwt2_ll(amplitude)
        freq_tokens = self.freq_input_proj(freq_input)
        freq_out = self.freq_mamba(freq_tokens) + freq_tokens
        freq_pooled = freq_out.mean(dim=1)

        return self.classifier(torch.cat([time_pooled, freq_pooled], dim=-1))
