"""Toque final de estúdio aplicado em pós-processamento determinístico.

Executa as correções objetivas dos guias diretamente nos pixels:
lineart preta → marrom RGB(61,43,40), sombras em cel shading com borda dura,
brancos limpos (esclerótica/Hi), remoção de granulado, +20% de saturação,
além de retículas de mangá, speed lines e folhas de settei.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from src.anime import CelShadingPipeline, SetteiSheetGenerator
from src.manga import MangaPanelLayout, ScreentoneConfig, apply_advanced_screentone, apply_speed_lines
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
    speed_lines: bool = False      # Linhas de velocidade de ação
    speed_line_density: float = 0.5
    panel_layout: bool = False     # Borda e calhas de painel de mangá
    panel_rows: int = 2
    panel_cols: int = 2
    settei_sheet: bool = False     # Formata em folha de produção de estúdio (settei sheet)


def _luminance(image: torch.Tensor) -> torch.Tensor:
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]


def _median3(image: torch.Tensor) -> torch.Tensor:
    patches = functional.unfold(functional.pad(image[None], (1, 1, 1, 1), mode="replicate"), 3)
    return patches.view(image.shape[0], 9, *image.shape[1:]).median(dim=1).values


def apply_studio_finish(image: torch.Tensor, config: FinishConfig | None = None) -> torch.Tensor:
    """``image``: (3, H, W) em [0, 1] → imagem corrigida no mesmo formato."""
    config = config or FinishConfig()
    image = image.float().clamp(0, 1)

    if config.speed_lines:
        image = apply_speed_lines(image, line_density=config.speed_line_density)

    if config.manga_mode:
        if config.denoise:
            image = _median3(image)
        luma = _luminance(image)
        if config.screentone:
            sc_config = ScreentoneConfig(frequency=20.0, angle=45.0, dot_type="round")
            image = apply_advanced_screentone(luma, sc_config)
        else:
            manga = torch.ones_like(luma)
            manga[luma < 0.35] = 0.0
            manga[(luma >= 0.35) & (luma < 0.70)] = 0.5
            image = manga.repeat(3, 1, 1)

        if config.panel_layout:
            image = MangaPanelLayout.draw_grid_gutters(image, rows=config.panel_rows, cols=config.panel_cols)
        return image.clamp(0, 1)

    if config.denoise:
        image = _median3(image)

    if config.cel_strength > 0:
        image = CelShadingPipeline.apply_cel_shading(image, num_levels=config.cel_levels)

    gray = _luminance(image)[None]
    image = (gray + (image - gray) * (1 + config.saturation_boost)).clamp(0, 1)

    high, low = image.max(dim=0).values, image.min(dim=0).values
    whites = (_luminance(image) > config.white_threshold) & ((high - low) < 0.1)
    image[:, whites] = 1.0

    luma = _luminance(image)
    dark = luma < config.lineart_threshold
    line = torch.tensor(LINEART_RGB, dtype=image.dtype, device=image.device)[:, None] / 255
    weight = 1 - luma[dark] / config.lineart_threshold
    image[:, dark] = image[:, dark] * (1 - weight) + line * weight

    if config.panel_layout:
        image = MangaPanelLayout.draw_grid_gutters(image, rows=config.panel_rows, cols=config.panel_cols)

    if config.settei_sheet:
        image = SetteiSheetGenerator.create_settei_layout(image)

    return image.clamp(0, 1)
