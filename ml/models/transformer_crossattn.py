"""OWN architecture (not from any cited paper): bidirectional multi-head cross-attention fusion between
amplitude and phase, instead of the ESP32 paper's late-fusion concat (transformer_dualbranch.py).

Each branch still self-attends independently first (same BranchEncoder as the dual-branch replica), but
then amplitude tokens query phase as key/value and vice versa, each through a full Transformer block
(cross-attention -> Add&Norm -> position-wise feed-forward -> Add&Norm, matching "Attention Is All You
Need"'s structure) -- so each channel's representation is actively reshaped by what the other channel
found, rather than the two streams only meeting once at the very end via concatenation. `num_layers`
stacks that block N times per direction; `norm_first` switches post-LN (original paper) vs pre-LN.
Motivation: ARGUS's channel ablation found amplitude alone is nearly as good as amplitude+phase+phase-
delta together (79.7% vs 78.9%), suggesting phase mostly adds a small correction on top of amplitude on
single-antenna hardware -- cross-attention lets amplitude tokens pull in exactly the phase information
relevant to each time step, rather than phase's contribution being averaged away across the whole
window before fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ml.models.common import BranchEncoder, CrossAttentionBlock


class CrossAttentionTransformer(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64,
                 dropout: float = 0.2, num_layers: int = 1, norm_first: bool = False):
        super().__init__()
        self.amp_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)
        self.phase_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)

        self.amp_attends_phase = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout, norm_first) for _ in range(num_layers)
        ])
        self.phase_attends_amp = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout, norm_first) for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes),
        )

    def _fuse(self, amplitude: torch.Tensor, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        amp_tokens = self.amp_branch(amplitude)      # (B, T, d_model)
        phase_tokens = self.phase_branch(phase)      # (B, T, d_model)

        for amp_block, phase_block in zip(self.amp_attends_phase, self.phase_attends_amp):
            amp_tokens, phase_tokens = amp_block(amp_tokens, phase_tokens), phase_block(phase_tokens, amp_tokens)
        return amp_tokens, phase_tokens

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_fused, phase_fused = self._fuse(amplitude, phase)
        fused = torch.cat([amp_fused.mean(dim=1), phase_fused.mean(dim=1)], dim=-1)
        return self.classifier(fused)

    def embed(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_fused, phase_fused = self._fuse(amplitude, phase)
        return torch.cat([amp_fused.mean(dim=1), phase_fused.mean(dim=1)], dim=-1)

    def attention_weights(self, amplitude: torch.Tensor, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        """For visualization/attention_rollout.py: raw cross-attention weights from the FIRST cross-
        attention layer (not used in the forward pass' return value, since nn.MultiheadAttention only
        exposes weights via need_weights)."""
        amp_tokens = self.amp_branch(amplitude)
        phase_tokens = self.phase_branch(phase)
        _, amp_to_phase_w = self.amp_attends_phase[0].cross_attn(
            amp_tokens, phase_tokens, phase_tokens, need_weights=True, average_attn_weights=False)
        _, phase_to_amp_w = self.phase_attends_amp[0].cross_attn(
            phase_tokens, amp_tokens, amp_tokens, need_weights=True, average_attn_weights=False)
        return {"amp_attends_phase": amp_to_phase_w, "phase_attends_amp": phase_to_amp_w}
