"""Minimal selective state-space (Mamba) block, pure PyTorch, sequential scan -- for CPU-only
environments where the official `mamba-ssm` package's custom CUDA kernels aren't usable. Follows the
standard Mamba formulation (Gu & Dao, 2023) with a simplified dt-projection (no low-rank dt_rank
factorization, just a direct per-channel Linear) for tractability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv - 1, groups=self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.B_proj = nn.Linear(self.d_inner, d_state)
        self.C_proj = nn.Linear(self.d_inner, d_state)

        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(A_log)  # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, d_model)
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        delta = F.softplus(self.dt_proj(x_conv))          # (B, T, d_inner)
        B_param = self.B_proj(x_conv)                      # (B, T, d_state)
        C_param = self.C_proj(x_conv)                       # (B, T, d_state)
        A = -torch.exp(self.A_log)                          # (d_inner, d_state)

        deltaA = torch.exp(delta.unsqueeze(-1) * A)                              # (B, T, d_inner, d_state)
        deltaBx = delta.unsqueeze(-1) * B_param.unsqueeze(2) * x_conv.unsqueeze(-1)  # (B, T, d_inner, d_state)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = deltaA[:, t] * h + deltaBx[:, t]
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(-1) + self.D * x_conv[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)

        y = y * F.silu(z)
        return self.out_proj(y)
