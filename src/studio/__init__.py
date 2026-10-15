"""Regras de estúdio, visão analítica e acabamento final para imagens de anime."""

from .critic import Check, StudioReport, StudioStyleCritic
from .finish import FinishConfig, apply_studio_finish
from .rules import (
    CORRECTIONS,
    FORBIDDEN_DOMAIN_KEYWORDS,
    NEGATIVE_PROMPT,
    PALETTES,
    STUDIO_PILLARS,
    build_negative_prompt,
    build_prompt,
    enforce_manga_anime_style,
    validate_manga_anime_domain,
)
from .visualization import render_terminal_image, save_html_gallery

__all__ = [
    "CORRECTIONS",
    "FORBIDDEN_DOMAIN_KEYWORDS",
    "Check",
    "FinishConfig",
    "NEGATIVE_PROMPT",
    "PALETTES",
    "STUDIO_PILLARS",
    "StudioReport",
    "StudioStyleCritic",
    "apply_studio_finish",
    "build_negative_prompt",
    "build_prompt",
    "enforce_manga_anime_style",
    "render_terminal_image",
    "save_html_gallery",
    "validate_manga_anime_domain",
]
