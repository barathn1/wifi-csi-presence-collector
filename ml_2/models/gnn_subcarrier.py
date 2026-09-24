"""GNN over subcarriers -- instead of assuming CNN-style local structure, model subcarriers as graph
nodes with a fixed (non-learned) banded adjacency by frequency proximity, and let 2 GCN layers mix
information across that graph. Node features are each subcarrier's own [amp_mean, amp_std, phase_mean,
phase_std] within the window -- cheap, and keeps every subcarrier's identity intact as a distinct node
rather than pooling across the band up front (the per-subcarrier-identity assumption this whole project
leans on).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def banded_adjacency(n_nodes: int, bandwidth: int = 4) -> torch.Tensor:
    """Symmetric, self-looped, row-normalized adjacency connecting each subcarrier to its `bandwidth`
    nearest neighbors on either side in frequency -- a fixed prior, not learned, since subcarrier
    ADJACENCY (which ones are near each other in frequency) is a known fact about the data, not
    something the model needs to discover."""
    idx = torch.arange(n_nodes)
    dist = (idx[:, None] - idx[None, :]).abs()
    adj = (dist <= bandwidth).float()
    deg = adj.sum(dim=-1, keepdim=True)
    return adj / deg.clamp(min=1.0)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # x: (B, N, in_dim)
        return F.relu(self.linear(adj @ x))


class SubcarrierGNN(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, hidden_dim: int = 32, bandwidth: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.register_buffer("adj", banded_adjacency(n_subcarriers, bandwidth))
        self.gcn1 = GCNLayer(4, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    @staticmethod
    def _node_features(x: torch.Tensor) -> torch.Tensor:  # x: (B, T, n_subcarriers) -> (B, n_subcarriers, 2)
        return torch.stack([x.mean(dim=1), x.std(dim=1)], dim=-1)

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        amp_feats = self._node_features(amplitude)      # (B, N, 2)
        phase_feats = self._node_features(phase)         # (B, N, 2)
        node_x = torch.cat([amp_feats, phase_feats], dim=-1)  # (B, N, 4)
        h = self.gcn1(node_x, self.adj)
        h = self.dropout(h)
        h = self.gcn2(h, self.adj)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)
