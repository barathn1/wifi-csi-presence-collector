"""Single-input BiLSTM for the walking-only cross-day identity task, matching the pasted spec's Keras
architecture 1:1 (stacked BiLSTM -> Dense -> 32-d embedding -> softmax head), in PyTorch since this
repo's `.venv-ml` has torch, not tensorflow. Input is the combined [batch, 400, 2k] tensor (k
amplitude + k phase channels per timestep, k=30 by default) produced by
`walk_bilstm_pipeline.apply_topk` + `per_window_zscore` -- not the two-branch (amp, phase) split
`ml/models/lstm.py`'s BiLSTMDualBranch uses elsewhere in this repo.

`recurrent_dropout` (the spec's Keras `LSTM(..., recurrent_dropout=0.3)`) has no direct PyTorch
`nn.LSTM` equivalent (cuDNN doesn't support per-gate recurrent dropout); `nn.LSTM`'s own `dropout=`
arg (applied between stacked layers, not recurrent connections) is used for the between-layer dropout
the spec's `dropout=0.3` arg maps to, and an explicit `nn.Dropout` after each BiLSTM's output stands in
for the recurrent-dropout regularization the Keras version gets for free.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMTriplet(nn.Module):
    def __init__(self, n_features: int, num_persons: int, lstm1_hidden: int = 128, lstm2_hidden: int = 64,
                 dropout: float = 0.3, dense_dropout: float = 0.4, embedding_dim: int = 32):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, lstm1_hidden, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(2 * lstm1_hidden, lstm2_hidden, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout)
        self.dense = nn.Sequential(nn.Linear(2 * lstm2_hidden, 64), nn.ReLU())
        self.dense_dropout = nn.Dropout(dense_dropout)
        self.embedding = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_persons)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 400, n_features). Returns (embedding [B, 32], logits [B, num_persons])."""
        out, _ = self.lstm1(x)               # (B, 400, 2*lstm1_hidden), return_sequences=True
        out = self.drop1(out)
        out, _ = self.lstm2(out)             # (B, 400, 2*lstm2_hidden)
        out = out[:, -1, :]                  # return_sequences=False: last timestep only
        out = self.drop2(out)
        out = self.dense(out)
        out = self.dense_dropout(out)
        embedding = self.embedding(out)      # linear activation, matches spec's "gait signature"
        logits = self.classifier(embedding)
        return embedding, logits
