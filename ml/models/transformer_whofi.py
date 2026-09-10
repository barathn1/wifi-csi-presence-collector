"""Replica of WhoFi's architecture (arXiv:2507.12869): a single 1-layer transformer over amplitude only
(their own ablation found 3 layers unstable/worse), mean-pooled and L2-normalized into an embedding
("signature module"), trained with a contrastive/in-batch-negative loss (see training/losses.py) for
verification rather than closed-set classification. A linear classification head is included too so
the same architecture can also report plain accuracy for tasks 0/A/B in the sweep.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.common import BranchEncoder


class WhoFiTransformer(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.amp_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout)
        self.signature_module = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def embed(self, amplitude: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        """L2-normalized embedding ('signature') -- phase is accepted but ignored, matching WhoFi's
        amplitude-only design; kept in the signature so every model in the zoo shares one call shape."""
        tokens = self.amp_branch(amplitude)
        pooled = tokens.mean(dim=1)
        signature = self.signature_module(pooled)
        return F.normalize(signature, p=2, dim=-1)

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.embed(amplitude))
