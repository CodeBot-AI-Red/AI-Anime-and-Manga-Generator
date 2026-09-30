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

    def step(
        self,
        predicted_noise: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        *,
        previous_timestep: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Take one DDIM-style reverse step, including when timesteps are skipped.

        DDPM's one-step posterior only applies to adjacent values.  Sampling
        with a small number of inference steps skips values, so reconstructing
        the clean image and moving directly to ``previous_timestep`` avoids an
        incorrect reverse process and makes the final step deterministic.
        """
        if not 0 <= timestep < self.num_timesteps:
            raise ValueError("timestep fora do intervalo do scheduler.")
        previous_timestep = timestep - 1 if previous_timestep is None else previous_timestep
        if not -1 <= previous_timestep < timestep:
            raise ValueError("previous_timestep deve ser menor que timestep.")
        alpha_bar = self.alpha_bars[timestep]
        previous_alpha_bar = self.alpha_bars[previous_timestep] if previous_timestep >= 0 else torch.ones_like(alpha_bar)
        predicted_clean = (sample - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        return previous_alpha_bar.sqrt() * predicted_clean + (1 - previous_alpha_bar).sqrt() * predicted_noise
