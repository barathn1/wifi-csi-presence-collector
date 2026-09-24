"""Small VAE trained on authorized-only feature vectors, used to synthesize near-boundary "almost
authorized" negatives for stress-testing/tightening an open-set threshold (OC-SVM, prototypical,
ArcFace) -- real stranger data is inherently scarce, so this manufactures harder negatives than random
noise would: decode latent points pushed slightly OUTSIDE the radius the authorized-only prior
actually occupies, rather than pure Gaussian noise in feature space.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.01 * kl


def train_vae(X_authorized: np.ndarray, latent_dim: int = 16, epochs: int = 100, lr: float = 1e-3,
              device: torch.device = torch.device("cpu")) -> FeatureVAE:
    vae = FeatureVAE(input_dim=X_authorized.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    X = torch.from_numpy(X_authorized.astype(np.float32)).to(device)
    for _ in range(epochs):
        recon, mu, logvar = vae(X)
        loss = vae_loss(recon, X, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return vae


@torch.no_grad()
def sample_hard_negatives(vae: FeatureVAE, n_samples: int, latent_dim: int, push_factor: float = 1.8,
                           device: torch.device = torch.device("cpu")) -> np.ndarray:
    """Samples latent points from the standard normal prior, then scales their NORM outward by
    `push_factor` (>1) before decoding -- pushes decoded samples toward the edge of / just past the
    authorized manifold's typical latent radius, i.e. "almost authorized but not quite", rather than
    generating typical/central authorized-like samples (push_factor=1.0 would just resample normal
    authorized-like data, not a hard negative)."""
    z = torch.randn(n_samples, latent_dim, device=device)
    z = z * push_factor
    return vae.decode(z).cpu().numpy()
