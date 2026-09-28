"""Deterministic DDPM-style noise schedule implemented locally."""

from __future__ import annotations

import torch


class NoiseScheduler:
    """Linear beta scheduler for forward noising and DDPM reverse sampling."""

    def __init__(self, num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu") -> None:
        if num_timesteps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("Parâmetros de scheduler inválidos.")
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean_images: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if clean_images.shape != noise.shape or timesteps.shape != (clean_images.shape[0],):
            raise ValueError("Imagem, ruído e timesteps incompatíveis.")
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * clean_images + (1 - alpha_bar).sqrt() * noise

    def step(self, predicted_noise: torch.Tensor, timestep: int, sample: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        alpha, alpha_bar, beta = self.alphas[timestep], self.alpha_bars[timestep], self.betas[timestep]
        mean = (sample - beta / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
        if timestep == 0:
            return mean
        variance = beta * (1 - self.alpha_bars[timestep - 1]) / (1 - alpha_bar)
        return mean + variance.sqrt() * torch.randn(sample.shape, device=sample.device, dtype=sample.dtype, generator=generator)
