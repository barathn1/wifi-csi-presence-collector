"""SimCLR-style contrastive self-supervised pretraining -- attacks the label-scarcity problem directly:
build positive pairs via augmentation (subcarrier dropout, amplitude jitter, time-shift) on UNLABELED
CSI windows (every window in the pool, not just labeled ones), train an embedding to be invariant to
those augmentations via NT-Xent, then a downstream task only needs a lightweight linear probe on top of
an already-good embedding instead of learning representations from scratch on a handful of labeled
sessions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def augment(amplitude: torch.Tensor, phase: torch.Tensor, dropout_p: float = 0.15,
            jitter_std: float = 0.05, max_time_shift: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """One random augmented view. `amplitude`/`phase`: (B, T, n_subcarriers)."""
    B, T, n_sub = amplitude.shape
    mask = (torch.rand(B, 1, n_sub, device=amplitude.device) > dropout_p).float()
    amp = amplitude * mask + torch.randn_like(amplitude) * jitter_std
    pha = phase * mask
    shift = torch.randint(-max_time_shift, max_time_shift + 1, (1,)).item()
    if shift != 0:
        amp = torch.roll(amp, shifts=shift, dims=1)
        pha = torch.roll(pha, shifts=shift, dims=1)
    return amp, pha


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Standard NT-Xent (SimCLR) contrastive loss over a batch of `2N` embeddings (N augmented pairs):
    each embedding's positive is its OWN pair's other view; every other embedding in the batch (both
    other originals and other augmented views) is a negative."""
    N = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)  # (2N, D)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)


def contrastive_pretrain_epoch(backbone: nn.Module, unlabeled_loader, opt: torch.optim.Optimizer,
                                device: torch.device, temperature: float = 0.2) -> float:
    backbone.train()
    total_loss, n = 0.0, 0
    for amp, phase, _ in unlabeled_loader:
        amp, phase = amp.to(device), phase.to(device)
        amp1, pha1 = augment(amp, phase)
        amp2, pha2 = augment(amp, phase)
        z1 = backbone(amp1, pha1)
        z2 = backbone(amp2, pha2)
        loss = nt_xent_loss(z1, z2, temperature)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


class LinearProbe(nn.Module):
    """Frozen-backbone downstream classifier -- the point of pretraining: only THIS needs to learn
    from the small labeled set."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, n_classes)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.linear(embedding)
