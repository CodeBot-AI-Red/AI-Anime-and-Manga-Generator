"""Autograd training adapter for the project diffusion denoiser."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as functional

from src.models.config import ModelConfig
from src.models.diffusion import DiffusionDenoiser
from src.models.scheduler import NoiseScheduler


class DiffusionTrainingModel:
    """Uses existing processed batches to learn to predict added image noise."""

    is_diffusion_model = True

    def __init__(self, model_config: ModelConfig | None = None, learning_rate: float = 1e-3, device: str = "cpu", seed: int = 42) -> None:
        self.config = model_config or ModelConfig()
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.denoiser = DiffusionDenoiser(self.config).to(self.device)
        self.scheduler = NoiseScheduler(self.config.num_timesteps, device=str(self.device))
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=learning_rate)
        self.seed = seed

    @staticmethod
    def prompts(metadata: tuple[object, ...]) -> list[str]:
        return [getattr(item, "caption", None) or getattr(item, "scene_description", None) or "imagem de anime" for item in metadata]

    def _images(self, batch: object) -> torch.Tensor:
        inputs = getattr(batch, "inputs")
        return torch.tensor(inputs.values, dtype=torch.float32, device=self.device).reshape(inputs.shape) * 2 - 1

    def loss_for_batch(self, batch: object) -> torch.Tensor:
        images = self._images(batch)
        timesteps = torch.randint(0, self.config.num_timesteps, (images.shape[0],), device=self.device)
        noise = torch.randn_like(images)
        # Classifier-free guidance teaches the same network an unconditional
        # path without requiring a second model or external text encoder.
        drop_conditioning = self.denoiser.training and torch.rand((), device=self.device).item() < self.config.condition_dropout
        prediction = self.denoiser(
            self.scheduler.add_noise(images, noise, timesteps), timesteps,
            self.prompts(getattr(batch, "metadata")), drop_conditioning=drop_conditioning,
        )
        return functional.mse_loss(prediction, noise)

    def train_step(self, batch: object) -> float:
        self.denoiser.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_for_batch(batch)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def validate_batch(self, batch: object) -> float:
        self.denoiser.eval()
        return float(self.loss_for_batch(batch).cpu())

    @torch.no_grad()
    def sample(self, prompts: list[str], sampling_steps: int | None = None, guidance_scale: float | None = None) -> torch.Tensor:
        if not prompts:
            raise ValueError("Forneça ao menos um prompt para gerar imagens.")
        self.denoiser.eval()
        steps = min(sampling_steps or self.config.generation_steps, self.config.num_timesteps)
        indices = torch.linspace(self.config.num_timesteps - 1, 0, steps, device=self.device).long().unique_consecutive()
        height, width = self.config.image_resolution[1], self.config.image_resolution[0]
        sample = torch.randn((len(prompts), self.config.image_channels, height, width), device=self.device)
        scale = self.config.guidance_scale if guidance_scale is None else guidance_scale
        if scale < 0:
            raise ValueError("guidance_scale não pode ser negativo.")
        for position, timestep in enumerate(indices):
            times = torch.full((len(prompts),), int(timestep), device=self.device, dtype=torch.long)
            conditional_noise = self.denoiser(sample, times, prompts)
            if scale:
                unconditional_noise = self.denoiser(sample, times, prompts, drop_conditioning=True)
                predicted_noise = unconditional_noise + scale * (conditional_noise - unconditional_noise)
            else:
                predicted_noise = conditional_noise
            previous = int(indices[position + 1]) if position + 1 < len(indices) else -1
            sample = self.scheduler.step(predicted_noise, int(timestep), sample, previous_timestep=previous)
        return ((sample.clamp(-1, 1) + 1) / 2).cpu()

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, float]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model": self.denoiser.state_dict(), "optimizer": self.optimizer.state_dict(), "metrics": metrics, "model_config": self.config.__dict__}, path)
        return path

    def load_checkpoint(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.denoiser.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return int(checkpoint["epoch"])
