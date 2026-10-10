"""Regras de estúdio, visão analítica e acabamento final para imagens de anime."""

from .critic import Check, StudioReport, StudioStyleCritic
from .finish import FinishConfig, apply_studio_finish
from .rules import CORRECTIONS, NEGATIVE_PROMPT, PALETTES, STUDIO_PILLARS, build_negative_prompt, build_prompt

__all__ = [
    "CORRECTIONS", "Check", "FinishConfig", "NEGATIVE_PROMPT", "PALETTES", "STUDIO_PILLARS",
    "StudioReport", "StudioStyleCritic", "apply_studio_finish", "build_negative_prompt", "build_prompt",
]
