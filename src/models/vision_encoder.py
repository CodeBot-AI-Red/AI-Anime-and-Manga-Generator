"""Encoder visual (ViT) treinado do zero que permite ao modelo "ver" imagens.

Transforma imagens de referência — enviadas pelo usuário ou geradas antes
pelo próprio modelo — em tokens visuais que entram na mesma atenção cruzada
do prompt. Assim a U-Net reaproveita personagem, paleta, pose e composição.
Nenhum peso pré-treinado ou API externa é usado.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional

from .config import ModelConfig


class VisionEncoder(nn.Module):
    """Vision Transformer: patches → tokens com a dimensão do condicionamento."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim, patch = config.time_embedding_dim, config.vision_patch_size
        self.config = config
        self.input_resolution = config.vision_input_resolution
        self.patches_per_side = self.input_resolution // patch
        self.patch_embedding = nn.Conv2d(config.image_channels, dim, kernel_size=patch, stride=patch)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patches_per_side ** 2 + 1, dim))
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.reference_embedding = nn.Embedding(config.max_reference_images, dim)
        layer = nn.TransformerEncoderLayer(
            dim, config.attention_heads, dim_feedforward=dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.vision_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    @property
    def tokens_per_image(self) -> int:
        return self.patches_per_side ** 2 + 1

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: (N, C, H, W) em [0, 1] → tokens (N, patches + 1, dim)."""
        if images.ndim != 4 or images.shape[1] != self.config.image_channels:
            raise ValueError("Referências devem ter a forma (N, canais, altura, largura).")
        size = (self.input_resolution, self.input_resolution)
        if tuple(images.shape[-2:]) != size:
            images = functional.interpolate(images, size=size, mode="bilinear", align_corners=False)
        hidden = self.patch_embedding(images * 2 - 1).flatten(2).transpose(1, 2)
        hidden = torch.cat((self.class_token.expand(hidden.shape[0], -1, -1), hidden), dim=1)
        return self.final_norm(self.encoder(hidden + self.position_embedding))

    def forward(self, references: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``references``: (B, R, C, H, W) → (tokens (B, R·T, dim), pooled (B, dim))."""
        if references.ndim != 5:
            raise ValueError("Referências devem ter a forma (lote, referências, canais, altura, largura).")
        batch, count = references.shape[:2]
        if count > self.config.max_reference_images:
            raise ValueError(f"No máximo {self.config.max_reference_images} imagens de referência por amostra.")
        tokens = self.encode_images(references.flatten(0, 1)).view(batch, count, self.tokens_per_image, -1)
        tokens = tokens + self.reference_embedding(torch.arange(count, device=references.device))[None, :, None, :]
        pooled = tokens[:, :, 0].mean(dim=1)
        return tokens.flatten(1, 2), pooled
