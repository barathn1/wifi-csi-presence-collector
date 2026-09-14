"""OWN architecture: the day's empty-room baseline (Variant A's amp_mean/std, phase_mean/std -- see
calibration.py) is fed in as 4 extra tokens the main window sequence cross-attends to, instead of being
subtracted from the input as hard preprocessing. The model gets to learn HOW to use the calibration
signal (e.g. attend to it more for some subcarriers than others) rather than always applying the same
fixed linear normalization everywhere. Each cross-attention step is a full Transformer block
(attention -> Add&Norm -> position-wise feed-forward -> Add&Norm, matching "Attention Is All You
Need"); `num_layers` stacks it N times, `norm_first` switches post-LN (original paper) vs pre-LN.

`set_calibration_context()` is called once per day (or once per fold/split) with that day's
DayBaseline; every window in a batch attends to the same context, matching how a real deployment would
compute one baseline each morning and reuse it all day.

Scale-mismatch fix (2026-09-11, see ml/reports/day2_next_steps.md item 4): the 4 context rows --
amp_mean, amp_std, phase_mean, phase_std -- come in on wildly different numeric scales (amplitude means
~40-90, phase values ~+-3 radians, a >20x gap) but were sharing one nn.Linear projection with no signal
telling the model which row is which. Two fixes applied: (1) each row is standardized to zero-mean/
unit-std across subcarriers before projection, so they arrive at a comparable scale; (2) a learned
per-token-type embedding (4 x d_model) is added after projection, giving the model an explicit "this is
the amp_mean token" signal instead of relying on residual scale differences to disambiguate.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.common import BranchEncoder, CrossAttentionBlock

EPS = 1e-6


class CalibrationContextTransformer(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64,
                 dropout: float = 0.2, num_layers: int = 1, norm_first: bool = False):
        super().__init__()
        self.amp_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.phase_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.context_proj = nn.Linear(n_subcarriers, d_model)
        self.context_type_embed = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)

        self.amp_attends_context = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout, norm_first) for _ in range(num_layers)
        ])
        self.phase_attends_context = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout, norm_first) for _ in range(num_layers)
        ])

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
        # standardize each of the 4 rows to zero-mean/unit-std across subcarriers -- puts amp/phase
        # mean/std on a comparable scale before they share context_proj's weights.
        ctx = (ctx - ctx.mean(dim=-1, keepdim=True)) / (ctx.std(dim=-1, keepdim=True) + EPS)
        self.context_raw = ctx.to(device) if device is not None else ctx

    def _fuse(self, amplitude: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = amplitude.shape[0]
        amp_tokens = self.amp_branch(amplitude)
        phase_tokens = self.phase_branch(phase)
        context_tokens = self.context_proj(self.context_raw.to(amplitude.device)) + self.context_type_embed
        context_tokens = context_tokens.expand(batch_size, -1, -1)

        for amp_block, phase_block in zip(self.amp_attends_context, self.phase_attends_context):
            amp_tokens = amp_block(amp_tokens, context_tokens)
            phase_tokens = phase_block(phase_tokens, context_tokens)
        return amp_tokens, phase_tokens

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_fused, phase_fused = self._fuse(amplitude, phase)
        fused = torch.cat([amp_fused.mean(dim=1), phase_fused.mean(dim=1)], dim=-1)
        return self.classifier(fused)

    def embed(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_fused, phase_fused = self._fuse(amplitude, phase)
        return torch.cat([amp_fused.mean(dim=1), phase_fused.mean(dim=1)], dim=-1)
