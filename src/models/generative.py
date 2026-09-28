"""Modelo generativo iterativo mínimo, local e preparado para evolução."""

from __future__ import annotations

from collections.abc import Sequence

from .config import ModelConfig
from .latent import LatentRepresentation
from .tensors import Tensor
from .text_conditioning import TextConditioner


class SmallGenerativeModel:
    """Prediz uma pequena atualização latente condicionada por texto e etapa."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.guidance_scale = 0.12

    def forward(self, latents: Tensor, text_embeddings: Tensor, generation_step: int) -> Tensor:
        if len(latents.shape) != 4 or latents.shape[1] != self.config.latent_channels:
            raise ValueError("Latentes incompatíveis com o modelo generativo.")
        if text_embeddings.shape != (latents.shape[0], self.config.text_embedding_dim):
            raise ValueError("Embeddings de texto incompatíveis com o lote latente.")
        if not 0 <= generation_step < self.config.generation_steps:
            raise ValueError("generation_step está fora do intervalo configurado.")
        batch, channels, height, width = latents.shape
        spatial_size = channels * height * width
        time_scale = (generation_step + 1) / self.config.generation_steps
        values = []
        for index, latent_value in enumerate(latents.values):
            batch_index = index // spatial_size
            channel_index = (index // (height * width)) % channels
            condition = text_embeddings.values[batch_index * self.config.text_embedding_dim + channel_index % self.config.text_embedding_dim]
            values.append((latent_value - condition) * self.guidance_scale * time_scale)
        return Tensor(latents.shape, tuple(values))


class IterativeImageGenerator:
    """Executa um pequeno processo de refinamento latente a partir de um prompt."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.text_conditioner = TextConditioner(self.config)
        self.latent_representation = LatentRepresentation(self.config)
        self.model = SmallGenerativeModel(self.config)

    def generate_latents(self, prompts: Sequence[str]) -> Tensor:
        embeddings = self.text_conditioner.encode(prompts)
        width, height = self.config.image_resolution
        latent_shape = (len(prompts), self.config.latent_channels,
                        height // self.config.downsample_factor, width // self.config.downsample_factor)
        initial = self._initial_latents(prompts, latent_shape)
        latents = initial
        for step in range(self.config.generation_steps):
            update = self.model.forward(latents, embeddings, step)
            latents = Tensor(latents.shape, tuple(value - correction for value, correction in zip(latents.values, update.values)))
        return latents

    def generate(self, prompt: str) -> Tensor:
        """Gera uma imagem BCHW unitária normalizada para um prompt simples."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt deve ser uma string não vazia.")
        return self.latent_representation.decode(self.generate_latents([prompt]))

    @staticmethod
    def _initial_latents(prompts: Sequence[str], shape: tuple[int, int, int, int]) -> Tensor:
        _, channels, height, width = shape
        values = tuple(((sum(ord(char) for char in prompt) + index * 17) % 101) / 100
                       for prompt in prompts for index in range(channels * height * width))
        return Tensor(shape, values)
