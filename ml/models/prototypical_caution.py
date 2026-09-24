"""CAUTION replica (Wang, Yang, Cui, Xie & Sun, "CAUTION: A Robust WiFi-Based Human Authentication
System via Few-Shot Open-Set Recognition," IEEE IoT-J vol.9 no.18, 2022): prototypical-network
few-shot identification + a distance-ratio intruder threshold, instead of a plain closed-set
classifier -- the point being that this is the one model in this project's zoo that is explicitly
*designed* to say "neither of the enrolled people," rather than being forced to pick one (see
data_pipeline/tasks.py's taskF docstring and RESEARCH_NOTES.md section 5's ARGUS/Trans-MAML citations
on why a closed-set softmax classifier structurally can't do that).

Algorithm (paper sections III-B/III-C):
1. Feature extractor F_theta maps a CSI window to a low-dimensional embedding.
2. Each enrolled user's prototype (CSI profile) c_k is the mean embedding of their support-set windows.
3. Training minimizes cross-entropy over a softmax of NEGATIVE distances to the prototypes
   (episodic support/query splits -- ml/training/train_caution_protonet.py).
4. At inference: rank the two nearest prototypes to a query embedding, take the distance ratio
   R = d(x, nearest) / d(x, second-nearest). Small R = confidently near one enrolled user and far from
   the other(s); R -> 1 = ambiguous/far from everyone, the paper's intruder signature. R is compared
   against a threshold T tuned using ONLY enrolled-user data (never real intruder data) -- see
   train_caution_protonet.py::calibrate_threshold.

With exactly 2 enrolled identities (anjali, barath) the "second-nearest" prototype is always "the
other one," which is a real, if simplified (K=2 instead of the paper's K=15), instantiation of the
same mechanism.

Encoder choice: the paper uses a 3-layer CNN. This replica reuses `ml.models.common.BranchEncoder`
(the amplitude-only transformer branch already used by `transformer_whofi.WhoFiTransformer`, this
project's own established best architecture per ml/reports/day2_next_steps.md's headline result)
instead of a fresh CNN -- RESEARCH_NOTES.md section 2 independently found a transformer beats a
CNN-only baseline on this same class of single-antenna ESP32 hardware (arXiv:2507.12854), so this is
a deliberate, documented substitution of the encoder, not a shortcut: CAUTION's actual algorithmic
contribution (prototypes + distance-ratio threshold) is unchanged and is what's being tested here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.common import BranchEncoder


class CautionEncoder(nn.Module):
    """amplitude-only branch -> mean-pooled, L2-normalized embedding. Same `embed(amp, phase)` call
    shape as every other model in the zoo (phase accepted but ignored, matching WhoFiTransformer)."""

    def __init__(self, n_subcarriers: int, d_model: int = 32, n_heads: int = 4, d_ff: int = 64,
                 dropout: float = 0.2, num_layers: int = 1, norm_first: bool = False):
        super().__init__()
        self.amp_branch = BranchEncoder(n_subcarriers, d_model, n_heads, d_ff, dropout, num_layers, norm_first)

    def embed(self, amplitude: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.amp_branch(amplitude)
        pooled = tokens.mean(dim=1)
        return F.normalize(pooled, p=2, dim=-1)

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        return self.embed(amplitude, phase)


def compute_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Paper eq.(1): c_k = mean embedding of class k's support-set windows."""
    return torch.stack([embeddings[labels == k].mean(dim=0) for k in range(n_classes)])


def prototypical_logits(query_emb: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Paper eq.(2)/(3): softmax over NEGATIVE squared-Euclidean distance to each prototype. Returns
    raw logits (-distance^2); caller applies softmax/cross-entropy."""
    dists_sq = torch.cdist(query_emb, prototypes, p=2) ** 2  # (B, n_classes)
    return -dists_sq


def distance_ratio(query_emb: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Paper eq.(4): R = d(x, nearest prototype) / d(x, second-nearest prototype)."""
    dists = torch.cdist(query_emb, prototypes, p=2)  # (B, n_classes)
    sorted_dists, _ = torch.sort(dists, dim=-1)
    d1, d2 = sorted_dists[:, 0], sorted_dists[:, 1]
    return d1 / (d2 + 1e-8)
