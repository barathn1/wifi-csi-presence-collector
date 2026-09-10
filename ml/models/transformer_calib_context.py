"""OWN architecture: the day's empty-room baseline (Variant A's amp_mean/std, phase_mean/std -- see
calibration.py) is fed in as 4 extra tokens the main window sequence cross-attends to, instead of being
subtracted from the input as hard preprocessing. The model gets to learn HOW to use the calibration
signal (e.g. attend to it more for some subcarriers than others) rather than always applying the same
fixed linear normalization everywhere.

`set_calibration_context()` is called once per day (or once per fold, in these Day-1 experiments) with
that day's DayBaseline; every window in a batch attends to the same context, matching how a real
deployment would compute one baseline each morning and reuse it all day.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.common import BranchEncoder


class CalibrationContextTransformer(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.amp_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout)
        self.phase_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout)
        self.context_proj = nn.Linear(n_subcarriers, d_model)

        self.amp_attends_context = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.phase_attends_context = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.amp_norm = nn.LayerNorm(d_model)
        self.phase_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes),
        )
        # (1, 4, n_subcarriers): [amp_mean, amp_std, phase_mean, phase_std] from that day's `none` sessions.
        self.register_buffer("context_raw", torch.zeros(1, 4, n_subcarriers), persistent=False)

    def set_calibration_context(self, amp_mean, amp_std, phase_mean, phase_std, device=None) -> None:
        ctx = torch.stack([
            torch.as_tensor(amp_mean, dtype=torch.float32),
            torch.as_tensor(amp_std, dtype=torch.float32),
            torch.as_tensor(phase_mean, dtype=torch.float32),
            torch.as_tensor(phase_std, dtype=torch.float32),
        ], dim=0).unsqueeze(0)
        self.context_raw = ctx.to(device) if device is not None else ctx

    def _fuse(self, amplitude: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = amplitude.shape[0]
        amp_tokens = self.amp_branch(amplitude)
        phase_tokens = self.phase_branch(phase)
        context_tokens = self.context_proj(self.context_raw.to(amplitude.device)).expand(batch_size, -1, -1)

        amp_ctx, _ = self.amp_attends_context(query=amp_tokens, key=context_tokens, value=context_tokens)
        phase_ctx, _ = self.phase_attends_context(query=phase_tokens, key=context_tokens, value=context_tokens)

        amp_fused = self.amp_norm(amp_tokens + amp_ctx)
        phase_fused = self.phase_norm(phase_tokens + phase_ctx)
        return amp_fused, phase_fused

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_fused, phase_fused = self._fuse(amplitude, phase)
        fused = torch.cat([amp_fused.mean(dim=1), phase_fused.mean(dim=1)], dim=-1)
        return self.classifier(fused)

    def embed(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_fused, phase_fused = self._fuse(amplitude, phase)
        return torch.cat([amp_fused.mean(dim=1), phase_fused.mean(dim=1)], dim=-1)
