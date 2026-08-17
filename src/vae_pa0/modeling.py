from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass
class VAEOutput:
    logits: torch.Tensor
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class MNISTVAE(nn.Module):
    """MLP VAE with a diagonal-Gaussian posterior and Bernoulli decoder."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.output_head = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x.flatten(1))
        return self.mu_head(hidden), self.logvar_head(hidden)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        standard_deviation = torch.exp(0.5 * logvar)
        if eps is None:
            eps = torch.randn_like(standard_deviation)
        return mu + standard_deviation * eps

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.decoder(z))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z))

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_logits(z)
        return VAEOutput(
            logits=logits,
            reconstruction=torch.sigmoid(logits),
            mu=mu,
            logvar=logvar,
            z=z,
        )

    def architecture(self) -> dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }


@dataclass
class ELBOLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


def negative_elbo(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> ELBOLoss:
    """Mean negative ELBO; BCE and KL are summed per image, then batch-averaged."""
    flat_target = target.flatten(1)
    reconstruction_per_image = functional.binary_cross_entropy_with_logits(
        logits, flat_target, reduction="none"
    ).sum(dim=1)
    kl_per_image = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)
    reconstruction = reconstruction_per_image.mean()
    kl = kl_per_image.mean()
    return ELBOLoss(
        total=reconstruction + beta * kl,
        reconstruction=reconstruction,
        kl=kl,
    )
