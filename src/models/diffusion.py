"""Small, from-scratch PyTorch diffusion components.

No pretrained weights, network calls, or external model APIs are used here.
The module is deliberately separate from the original lightweight ``Tensor``
contracts so those interfaces remain available while real autograd training is
provided by PyTorch when the declared dependency is installed.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

from .config import ModelConfig


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep features followed by a learnable projection."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4:
            raise ValueError("time_embedding_dim deve ser ao menos 4.")
        self.dimension = dimension
        self.projection = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, dimension))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps deve ter a forma (lote,).")
        half = self.dimension // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
        values = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((values.sin(), values.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = functional.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return self.projection(embedding)


class PromptEncoder(nn.Module):
    """Trainable byte-level prompt encoder, initialized from scratch."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def tokenize(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("Forneça um lote não vazio de prompts em texto.")
        encoded = [list(prompt.encode("utf-8")[:128]) or [0] for prompt in prompts]
        tokens = torch.zeros((len(encoded), max(map(len, encoded))), dtype=torch.long, device=device)
        for index, row in enumerate(encoded):
            tokens[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device) % self.vocab_size
        return tokens

    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenize(prompts, device)
        # Padding must not influence a caption embedding.  This matters for
        # mixed batches, where short prompts otherwise receive a large number
        # of learned ``0`` token embeddings.
        mask = (tokens != 0).unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp_min(1)
        return (self.embedding(tokens) * mask).sum(dim=1) / lengths


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(functional.silu(self.norm1(inputs)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(inputs) + self.conv2(functional.silu(hidden))


class DiffusionDenoiser(nn.Module):
    """Compact conditional U-Net that predicts noise in BCHW image space."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        channels, condition_dim = config.base_channels, config.time_embedding_dim
        self.time_embedding = TimestepEmbedding(condition_dim)
        self.prompt_encoder = PromptEncoder(config.text_vocab_size, condition_dim)
        self.condition = nn.Sequential(nn.Linear(condition_dim * 2, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.input = nn.Conv2d(config.image_channels, channels, 3, padding=1)
        self.down_block = ResidualBlock(channels, channels, condition_dim)
        self.downsample = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1)
        self.middle = ResidualBlock(channels * 2, channels * 2, condition_dim)
        self.upsample = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.up_block = ResidualBlock(channels * 2, channels, condition_dim)
        self.output = nn.Conv2d(channels, config.image_channels, 3, padding=1)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        prompts: list[str],
        *,
        drop_conditioning: bool = False,
    ) -> torch.Tensor:
        if noisy_images.ndim != 4 or noisy_images.shape[1] != self.config.image_channels:
            raise ValueError("Imagens devem ter a forma (lote, image_channels, altura, largura).")
        if noisy_images.shape[0] != len(prompts) or timesteps.shape != (noisy_images.shape[0],):
            raise ValueError("Imagem, timestep e prompt devem ter o mesmo tamanho de lote.")
        prompt_embedding = self.prompt_encoder(prompts, noisy_images.device)
        if drop_conditioning:
            prompt_embedding = torch.zeros_like(prompt_embedding)
        condition = self.condition(torch.cat((self.time_embedding(timesteps), prompt_embedding), dim=1))
        skip = self.down_block(self.input(noisy_images), condition)
        hidden = self.middle(self.downsample(skip), condition)
        upsampled = self.upsample(hidden)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = functional.interpolate(upsampled, size=skip.shape[-2:], mode="nearest")
        return self.output(self.up_block(torch.cat((upsampled, skip), dim=1), condition))
