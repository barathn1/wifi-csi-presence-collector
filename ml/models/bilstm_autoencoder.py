"""One-class BiLSTM autoencoder: trained on ONLY authorized walking windows (no unauthorized data,
no labels at all), to test whether reconstruction error -- not embedding distance, not softmax
confidence -- separates authorized from unauthorized walkers. Every classifier-style approach tried in
`ml/training/run_walk_bilstm_*openset.py` tries to separate people IN THE SAME learned space; this
instead asks the model to learn "what authorized reconstructs well," never showing it what
"unauthorized" looks like at all."""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int = 60, hidden: int = 64, bottleneck_dim: int = 32, seq_len: int = 400):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.bottleneck = nn.Linear(2 * hidden, bottleneck_dim)
        self.decoder_lstm = nn.LSTM(bottleneck_dim, hidden, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(2 * hidden, n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 400, n_features). Returns (reconstruction [B, 400, n_features], bottleneck [B, 32])."""
        out, _ = self.encoder_lstm(x)
        last = out[:, -1, :]                                  # return_sequences=False on the encoder
        z = self.bottleneck(last)
        z_repeated = z.unsqueeze(1).expand(-1, self.seq_len, -1)  # RepeatVector(seq_len)
        dec_out, _ = self.decoder_lstm(z_repeated)
        recon = self.output_layer(dec_out)
        return recon, z
