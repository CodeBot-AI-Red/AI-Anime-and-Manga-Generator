"""Autograd training adapter for the project diffusion denoiser."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as functional

from src.models.config import ModelConfig
from src.models.diffusion import DiffusionDenoiser
from src.models.scheduler import NoiseScheduler


class EMAModel:
    """Exponential Moving Average de pesos do modelo para amostragem suave de alta qualidade."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param.detach()))

    def copy_to(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.copy_(self.shadow[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, tensor in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)


class DiffusionTrainingModel:
    """Uses existing processed batches to learn to predict added image noise."""

    is_diffusion_model = True

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
        use_ema: bool = True,
        use_amp: bool = False,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.config = model_config or ModelConfig()
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.denoiser = DiffusionDenoiser(self.config).to(self.device)
        self.scheduler = NoiseScheduler(self.config.num_timesteps, device=str(self.device))
        self.optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=learning_rate)
        self.ema = EMAModel(self.denoiser) if use_ema else None
        self.use_amp = use_amp
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.accumulation_counter = 0
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
        # Drop captions independently, rather than discarding a whole batch.
        # This provides conditional and unconditional examples in every mixed batch.
        conditioning_mask = (torch.rand(images.shape[0], device=self.device) >= self.config.condition_dropout).to(images.dtype)
        # Ensina o encoder visual a "ver": em parte dos passos a própria imagem
        # limpa (reduzida) vira referência, aprendendo a reaproveitar paleta,
        # personagem e composição de uma imagem de referência.
        references = None
        if self.denoiser.vision_encoder is not None and torch.rand(()) >= self.config.reference_dropout:
            references = functional.interpolate((images + 1) / 2, scale_factor=0.5, mode="bilinear", align_corners=False).unsqueeze(1)
        prediction = self.denoiser(
            self.scheduler.add_noise(images, noise, timesteps), timesteps,
            self.prompts(getattr(batch, "metadata")), conditioning_mask=conditioning_mask,
            reference_images=references,
        )
        return functional.mse_loss(prediction, noise)

    def train_step(self, batch: object) -> float:
        self.denoiser.train()
        self.accumulation_counter += 1
        is_accumulating = (self.accumulation_counter % self.gradient_accumulation_steps) != 0

        if self.accumulation_counter % self.gradient_accumulation_steps == 1 or self.gradient_accumulation_steps == 1:
            self.optimizer.zero_grad(set_to_none=True)

        device_type = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        if self.use_amp and self.device.type in ("cuda", "cpu"):
            dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16
            with torch.autocast(device_type=device_type, dtype=dtype):
                loss = self.loss_for_batch(batch) / self.gradient_accumulation_steps
        else:
            loss = self.loss_for_batch(batch) / self.gradient_accumulation_steps

        loss.backward()

        if not is_accumulating:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.denoiser)

        return float((loss * self.gradient_accumulation_steps).detach().cpu())

    @torch.no_grad()
    def validate_batch(self, batch: object) -> float:
        self.denoiser.eval()
        return float(self.loss_for_batch(batch).cpu())

    @torch.no_grad()
    def sample(
        self,
        prompts: list[str],
        sampling_steps: int | None = None,
        guidance_scale: float | None = None,
        *,
        negative_prompts: list[str] | None = None,
        reference_images: torch.Tensor | None = None,
        init_images: torch.Tensor | None = None,
        strength: float = 0.6,
        dynamic_thresholding: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Gera imagens em [0, 1].

        - ``negative_prompts``: o que evitar (CFG negativo).
        - ``reference_images``: (B, R, C, H, W) em [0, 1] que o modelo "vê".
        - ``init_images``: (B, C, H, W) em [0, 1] para corrigir/refazer uma
          imagem existente (img2img); ``strength`` em (0, 1] controla quanto muda.
        """
        if not prompts:
            raise ValueError("Forneça ao menos um prompt para gerar imagens.")
        if not 0 < strength <= 1:
            raise ValueError("strength deve estar no intervalo (0, 1].")
        self.denoiser.eval()
        steps = min(sampling_steps or self.config.generation_steps, self.config.num_timesteps)
        start = self.config.num_timesteps - 1
        height, width = self.config.image_resolution[1], self.config.image_resolution[0]
        shape = (len(prompts), self.config.image_channels, height, width)
        noise = torch.randn(shape, device=self.device, generator=generator)
        if init_images is not None:
            init = init_images.to(self.device, torch.float32)
            if tuple(init.shape[-2:]) != (height, width):
                init = functional.interpolate(init, size=(height, width), mode="bilinear", align_corners=False)
            start = max(1, int(round(strength * (self.config.num_timesteps - 1))))
            steps = max(1, int(round(steps * strength)))
            sample = self.scheduler.add_noise(init * 2 - 1, noise, torch.full((len(prompts),), start, device=self.device, dtype=torch.long))
        else:
            sample = noise
        if reference_images is not None:
            reference_images = reference_images.to(self.device, torch.float32)
        indices = torch.linspace(start, 0, steps, device=self.device).long().unique_consecutive()
        scale = self.config.guidance_scale if guidance_scale is None else guidance_scale
        if scale < 0:
            raise ValueError("guidance_scale não pode ser negativo.")
        for position, timestep in enumerate(indices):
            times = torch.full((len(prompts),), int(timestep), device=self.device, dtype=torch.long)
            conditional_noise = self.denoiser(sample, times, prompts, reference_images=reference_images)
            if scale:
                unconditional_noise = self.denoiser(sample, times, prompts, drop_conditioning=True, negative_prompts=negative_prompts)
                predicted_noise = unconditional_noise + scale * (conditional_noise - unconditional_noise)
                if dynamic_thresholding and predicted_noise.abs().max() > 1.0:
                    s = torch.quantile(predicted_noise.abs().flatten(1), 0.995, dim=1).clamp_min(1.0)
                    predicted_noise = predicted_noise.clamp(-s[:, None, None, None], s[:, None, None, None])
            else:
                predicted_noise = conditional_noise
            previous = int(indices[position + 1]) if position + 1 < len(indices) else -1
            sample = self.scheduler.step(predicted_noise, int(timestep), sample, previous_timestep=previous)
        return ((sample.clamp(-1, 1) + 1) / 2).cpu()

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, float]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "epoch": epoch,
            "model": self.denoiser.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
            "model_config": self.config.__dict__,
        }
        if self.ema is not None:
            data["ema"] = self.ema.state_dict()
        torch.save(data, path)
        return path

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu") -> "DiffusionTrainingModel":
        """Recria a arquitetura exata salva no checkpoint e carrega os pesos."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(ModelConfig.from_dict(checkpoint.get("model_config", {})), device=device)
        model.load_checkpoint(path)
        return model

    def load_checkpoint(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.denoiser.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        return int(checkpoint["epoch"])
