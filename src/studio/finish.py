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
    manga_mode: bool = False       # Converte em arte de mangá preto e branco com retícula
    screentone: bool = True        # Aplica padrão de retícula (halftone) em meio-tom


def _luminance(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]


def _median3(image: torch.Tensor) -> torch.Tensor:
    patches = functional.unfold(functional.pad(image[None], (1, 1, 1, 1), mode="replicate"), 3)
    return patches.view(image.shape[0], 9, *image.shape[1:]).median(dim=1).values


def _apply_manga_screentone(luma: torch.Tensor, use_screentone: bool = True) -> torch.Tensor:
    height, width = luma.shape
    if not use_screentone:
        # Binarização/quantização direta de página de mangá
        manga = torch.ones_like(luma)
        manga[luma < 0.35] = 0.0
        manga[(luma >= 0.35) & (luma < 0.70)] = 0.5
        return manga.repeat(3, 1, 1)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=luma.device),
        torch.arange(width, device=luma.device),
        indexing="ij",
    )
    # Matriz de retícula (dot pattern halftone de 4x4)
    dot_pattern = (((xs % 4 - 1.5).pow(2) + (ys % 4 - 1.5).pow(2)) / 4.5).clamp(0, 1)

    manga_mono = torch.ones_like(luma)
    midtone_mask = (luma >= 0.25) & (luma < 0.75)
    # Quanto mais escuro o tom original, maior o ponto de retícula
    tone_threshold = (luma - 0.25) / 0.50
    manga_mono[midtone_mask] = (dot_pattern[midtone_mask] < tone_threshold[midtone_mask]).float() * 0.6 + 0.2
    manga_mono[luma < 0.25] = 0.0
    manga_mono[luma >= 0.75] = 1.0

    return manga_mono.repeat(3, 1, 1)


def apply_studio_finish(image: torch.Tensor, config: FinishConfig | None = None) -> torch.Tensor:
    """``image``: (3, H, W) em [0, 1] → imagem corrigida no mesmo formato."""
    config = config or FinishConfig()
    image = image.float().clamp(0, 1)

    if config.manga_mode:
        if config.denoise:
            image = _median3(image)
        luma = _luminance(image)
        return _apply_manga_screentone(luma, config.screentone).clamp(0, 1)

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
