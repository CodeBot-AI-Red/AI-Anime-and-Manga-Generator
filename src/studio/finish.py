"""Toque final de estúdio aplicado em pós-processamento determinístico.

Executa as correções objetivas dos guias diretamente nos pixels:
lineart preta → marrom RGB(61,43,40), sombras em cel shading com borda dura,
brancos limpos (esclerótica/Hi), remoção de granulado e +20% de saturação.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from .rules import LINEART_RGB


@dataclass(frozen=True)
class FinishConfig:
    cel_levels: int = 3            # Hi / Normal / 1ª sombra
    cel_strength: float = 0.6      # 0 desliga a quantização de sombras
    saturation_boost: float = 0.2  # +20% (guia: 20–25%)
    lineart_threshold: float = 0.2
    white_threshold: float = 0.86
    denoise: bool = True


def _luminance(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]


def _median3(image: torch.Tensor) -> torch.Tensor:
    patches = functional.unfold(functional.pad(image[None], (1, 1, 1, 1), mode="replicate"), 3)
    return patches.view(image.shape[0], 9, *image.shape[1:]).median(dim=1).values


def apply_studio_finish(image: torch.Tensor, config: FinishConfig | None = None) -> torch.Tensor:
    """``image``: (3, H, W) em [0, 1] → imagem corrigida no mesmo formato."""
    config = config or FinishConfig()
    image = image.float().clamp(0, 1)
    if config.denoise:
        image = _median3(image)

    luma = _luminance(image)
    if config.cel_strength > 0:
        levels = config.cel_levels
        stepped = ((luma * levels).floor().clamp(max=levels - 1) + 0.5) / levels
        target = luma + (stepped - luma) * config.cel_strength
        image = (image * (target / luma.clamp_min(1e-4))[None]).clamp(0, 1)

    gray = _luminance(image)[None]
    image = (gray + (image - gray) * (1 + config.saturation_boost)).clamp(0, 1)

    high, low = image.max(dim=0).values, image.min(dim=0).values
    whites = (_luminance(image) > config.white_threshold) & ((high - low) < 0.1)
    image[:, whites] = 1.0

    luma = _luminance(image)
    dark = luma < config.lineart_threshold
    line = torch.tensor(LINEART_RGB, dtype=image.dtype)[:, None] / 255
    weight = 1 - luma[dark] / config.lineart_threshold
    image[:, dark] = image[:, dark] * (1 - weight) + line * weight
    return image.clamp(0, 1)
