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
    """Trainable byte-level Transformer encoder, initialized from scratch."""

    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int = 128, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.vocab_size, self.max_length = vocab_size, max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, heads, dim_feedforward=embedding_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(embedding_dim)

    def tokenize(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("Forneça um lote não vazio de prompts em texto.")
        # Reserve token 0 for padding, so an empty prompt has one valid token.
        encoded = [[value % (self.vocab_size - 1) + 1 for value in prompt.encode("utf-8")[:self.max_length]] or [1] for prompt in prompts]
        tokens = torch.zeros((len(encoded), max(map(len, encoded))), dtype=torch.long, device=device)
        for index, row in enumerate(encoded):
            tokens[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device) % self.vocab_size
        return tokens

    def encode_tokens(self, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenize(prompts, device)
        valid = tokens != 0
        positions = torch.arange(tokens.shape[1], device=device).unsqueeze(0)
        hidden = self.embedding(tokens) + self.position_embedding(positions)
        return self.final_norm(self.encoder(hidden, src_key_padding_mask=~valid)), valid

    def forward(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        tokens, valid = self.encode_tokens(prompts, device)
        mask = valid.unsqueeze(-1)
        return (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


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


class CrossAttention(nn.Module):
    """Lets spatial image features select relevant tokens from each prompt."""

    def __init__(self, channels: int, condition_dim: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Canais de atenção devem ser divisíveis pelo número de cabeças.")
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(condition_dim, channels)
        self.value = nn.Linear(condition_dim, channels)
        self.output = nn.Linear(channels, channels)
        self.heads = heads

    def forward(self, inputs: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        spatial = self.norm(inputs).flatten(2).transpose(1, 2)
        head_dim = channels // self.heads
        query = self.query(spatial).view(batch, -1, self.heads, head_dim).transpose(1, 2)
        key = self.key(tokens).view(batch, -1, self.heads, head_dim).transpose(1, 2)
        value = self.value(tokens).view(batch, -1, self.heads, head_dim).transpose(1, 2)
        weights = (query @ key.transpose(-1, -2)) * (head_dim ** -0.5)
        weights = weights.masked_fill(~valid[:, None, None, :], torch.finfo(weights.dtype).min)
        attended = (weights.softmax(dim=-1) @ value).transpose(1, 2).reshape(batch, height * width, channels)
        return inputs + self.output(attended).transpose(1, 2).reshape(batch, channels, height, width)


class DiffusionDenoiser(nn.Module):
    """Compact conditional U-Net that predicts noise in BCHW image space."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        required_factor = config.downsample_factor ** 2
        if any(dimension % required_factor for dimension in config.image_resolution):
            raise ValueError("A U-Net de difusão exige image_resolution divisível por downsample_factor ao quadrado.")
        self.config = config
        channels, condition_dim = config.base_channels, config.time_embedding_dim
        self.time_embedding = TimestepEmbedding(condition_dim)
        self.prompt_encoder = PromptEncoder(config.text_vocab_size, condition_dim, config.max_prompt_length, config.text_encoder_layers, config.attention_heads)
        self.condition = nn.Sequential(nn.Linear(condition_dim * 2, condition_dim), nn.SiLU(), nn.Linear(condition_dim, condition_dim))
        self.input = nn.Conv2d(config.image_channels, channels, 3, padding=1)
        self.down_blocks = nn.ModuleList([ResidualBlock(channels, channels, condition_dim) for _ in range(config.unet_res_blocks)])
        self.downsample = nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1)
        self.down_blocks_deep = nn.ModuleList([ResidualBlock(channels * 2, channels * 2, condition_dim) for _ in range(config.unet_res_blocks)])
        self.downsample_deep = nn.Conv2d(channels * 2, channels * 4, 4, stride=2, padding=1)
        self.middle = nn.ModuleList([ResidualBlock(channels * 4, channels * 4, condition_dim) for _ in range(config.unet_res_blocks)])
        self.cross_attention = CrossAttention(channels * 4, condition_dim, config.attention_heads)
        self.upsample_deep = nn.ConvTranspose2d(channels * 4, channels * 2, 4, stride=2, padding=1)
        self.up_blocks_deep = nn.ModuleList([ResidualBlock(channels * 4 if index == 0 else channels * 2, channels * 2, condition_dim) for index in range(config.unet_res_blocks)])
        self.upsample = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.up_blocks = nn.ModuleList([ResidualBlock(channels * 2 if index == 0 else channels, channels, condition_dim) for index in range(config.unet_res_blocks)])
        self.output = nn.Sequential(nn.GroupNorm(min(8, channels), channels), nn.SiLU(), nn.Conv2d(channels, config.image_channels, 3, padding=1))

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        prompts: list[str],
        *,
        drop_conditioning: bool = False,
        conditioning_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_images.ndim != 4 or noisy_images.shape[1] != self.config.image_channels:
            raise ValueError("Imagens devem ter a forma (lote, image_channels, altura, largura).")
        if noisy_images.shape[0] != len(prompts) or timesteps.shape != (noisy_images.shape[0],):
            raise ValueError("Imagem, timestep e prompt devem ter o mesmo tamanho de lote.")
        prompt_tokens, token_valid = self.prompt_encoder.encode_tokens(prompts, noisy_images.device)
        prompt_embedding = (prompt_tokens * token_valid.unsqueeze(-1)).sum(dim=1) / token_valid.sum(dim=1, keepdim=True).clamp_min(1)
        if drop_conditioning:
            prompt_embedding = torch.zeros_like(prompt_embedding)
            prompt_tokens = torch.zeros_like(prompt_tokens)
        elif conditioning_mask is not None:
            if conditioning_mask.shape != (noisy_images.shape[0],):
                raise ValueError("conditioning_mask deve ter a forma (lote,).")
            prompt_embedding = prompt_embedding * conditioning_mask[:, None]
            prompt_tokens = prompt_tokens * conditioning_mask[:, None, None]
        condition = self.condition(torch.cat((self.time_embedding(timesteps), prompt_embedding), dim=1))
        skip = self.input(noisy_images)
        for block in self.down_blocks:
            skip = block(skip, condition)
        deep_skip = self.downsample(skip)
        for block in self.down_blocks_deep:
            deep_skip = block(deep_skip, condition)
        hidden = self.downsample_deep(deep_skip)
        for block in self.middle:
            hidden = block(hidden, condition)
        hidden = self.cross_attention(hidden, prompt_tokens, token_valid)
        hidden = self.upsample_deep(hidden)
        if hidden.shape[-2:] != deep_skip.shape[-2:]:
            hidden = functional.interpolate(hidden, size=deep_skip.shape[-2:], mode="nearest")
        hidden = torch.cat((hidden, deep_skip), dim=1)
        for block in self.up_blocks_deep:
            hidden = block(hidden, condition)
        hidden = self.upsample(hidden)
        if hidden.shape[-2:] != skip.shape[-2:]:
            hidden = functional.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
        hidden = torch.cat((hidden, skip), dim=1)
        for block in self.up_blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the exact number of parameters optimized by the local model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
