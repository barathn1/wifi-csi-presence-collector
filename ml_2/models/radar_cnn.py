"""Micro-Doppler-radar-inspired CNN: unlike every other model here, which feeds raw per-subcarrier
amplitude/phase time series directly, this one first takes an STFT of the amplitude signal (averaged
across subcarriers) ALONG THE TIME AXIS -- turning the window into a time-frequency spectrogram whose
low-frequency content (~1-2 Hz) is the actual gait-cadence signature RF/radar gait-identification
literature relies on, borrowed here since that literature is far more mature than WiFi CSI for this
exact problem (classify a walking human from a reflected RF signal).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RadarInspiredCNN(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, n_fft: int = 32, hop_length: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        del n_subcarriers  # amplitude is averaged across subcarriers before the STFT, see forward()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(32 * 4 * 4, n_classes)

    def _spectrogram(self, amplitude: torch.Tensor) -> torch.Tensor:  # (B, T, n_subcarriers) -> (B, F, T')
        signal = amplitude.mean(dim=-1)  # (B, T) -- micro-Doppler proxy: gait-induced amplitude modulation
        spec = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                           return_complex=True, center=True)
        return torch.log1p(spec.abs())  # (B, n_fft//2+1, n_frames)

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        del phase  # micro-Doppler signature is carried by amplitude modulation, not phase, here
        spec = self._spectrogram(amplitude).unsqueeze(1)  # (B, 1, F, T')
        h = self.conv(spec)
        h = self.dropout(h.flatten(1))
        return self.classifier(h)
