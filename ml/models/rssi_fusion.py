"""Fuse RSSI (a single frequency-agnostic scalar per packet) into any existing model in the zoo, as an
auxiliary signal on top of whatever amplitude/phase representation it already learns -- see
ml/reports/day2_next_steps.md item 7. RSSI is untouched by the Day1/Day2 channel-frequency mismatch
(see [[project-day2-cross-channel-root-cause]] in memory) in the sense that it isn't tied to any
subcarrier's identity, but it does drift across days on its own (confirmed directly on this data), so
it's calibrated the same way amplitude/phase are (`calibration.py::compute_day_rssi_baseline`) before
being fused, not used raw.

Implemented as a wrapper around any base model that exposes `.embed(amplitude, phase) -> (B, embed_dim)`
(every model in the transformer/CNN/LSTM family already does) rather than editing every model's
forward() signature -- keeps every existing script/model file untouched.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class RSSIFusionClassifier(nn.Module):
    def __init__(self, base_model: nn.Module, embed_dim: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.base_model = base_model
        # +2: [rssi_mean, rssi_std] per window, calibrated (per-day z-score, see _RSSIStatsDataset).
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 2, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, n_classes),
        )

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor, rssi_stats: torch.Tensor) -> torch.Tensor:
        embed = self.base_model.embed(amplitude, phase)
        return self.classifier(torch.cat([embed, rssi_stats], dim=-1))


class _RSSIStatsDataset(Dataset):
    """Wraps a CsiWindowDataset(..., include_rssi=True) and replaces the raw per-packet RSSI with its
    per-window [mean, std] under that row's OWN date's calibration -- lazy, one item at a time, same
    on-demand design as the base dataset (never materializes the full set in RAM)."""

    def __init__(self, base_ds, rssi_baseline_by_date: dict):
        self.base_ds = base_ds
        self.rssi_baseline_by_date = rssi_baseline_by_date

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, i: int):
        amp, phase, rssi, label = self.base_ds[i]
        date = self.base_ds.index.iloc[i]["date"]
        mean, std = self.rssi_baseline_by_date[date]
        z = (rssi - mean) / (std + 1e-6)
        rssi_stats = torch.stack([z.mean(), z.std()])
        return amp, phase, rssi_stats, label


def train_rssi_fusion_classifier(model: RSSIFusionClassifier, train_ds, test_ds, rssi_baseline_by_date: dict,
                                  epochs: int = 4, batch_size: int = 64, lr: float = 1e-3, seed: int = 0,
                                  device: torch.device = torch.device("cpu")) -> dict:
    """Same shape as ml/training/train.py::train_classifier, but for datasets built with
    CsiWindowDataset(..., include_rssi=True) and a model that also needs calibrated RSSI stats.
    `rssi_baseline_by_date`: {date: (mean, std)} from calibration.py::compute_day_rssi_baseline."""
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(_RSSIStatsDataset(train_ds, rssi_baseline_by_date),
                               batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(_RSSIStatsDataset(test_ds, rssi_baseline_by_date),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    for _ in range(epochs):
        model.train()
        for amp, phase, rssi_stats, label in train_loader:
            amp, phase, rssi_stats, label = amp.to(device), phase.to(device), rssi_stats.to(device), label.to(device)
            opt.zero_grad()
            loss = loss_fn(model(amp, phase, rssi_stats), label)
            loss.backward()
            opt.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for amp, phase, rssi_stats, label in test_loader:
            amp, phase, rssi_stats, label = amp.to(device), phase.to(device), rssi_stats.to(device), label.to(device)
            pred = model(amp, phase, rssi_stats).argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += len(label)

    return {"accuracy": correct / max(total, 1), "n_train": len(train_ds), "n_test": len(test_ds)}
