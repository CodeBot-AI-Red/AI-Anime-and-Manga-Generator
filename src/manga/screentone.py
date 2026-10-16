"""Screentone and halftone generator for professional manga shading.

Provides halftone dot patterns, cross-hatching, and line screening.
"""

from __future__ import annotations

import math
import torch


class ScreentoneConfig:
    """Configuration for manga screentone halftone generation."""

    def __init__(
        self,
        frequency: float = 16.0,  # Dots per scale unit (LPI)
        angle: float = 45.0,     # Standard 45 degree angle for halftone
        dot_type: str = "round",  # round, line, or crosshatch
        contrast: float = 1.0,
    ) -> None:
        if dot_type not in ("round", "line", "crosshatch"):
            raise ValueError(f"Unknown dot_type '{dot_type}'. Options: round, line, crosshatch")
        self.frequency = frequency
        self.angle = angle
        self.dot_type = dot_type
        self.contrast = contrast


def generate_screentone_pattern(
    shape: tuple[int, int],
    config: ScreentoneConfig | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generates a 2D halftone screentone pattern mask in range [0, 1]."""
    config = config or ScreentoneConfig()
    height, width = shape

    ys = torch.linspace(0, 1, height, device=device)
    xs = torch.linspace(0, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # Rotate coordinates by angle
    rad = math.radians(config.angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot_x = grid_x * cos_a - grid_y * sin_a
    rot_y = grid_x * sin_a + grid_y * cos_a

    if config.dot_type == "round":
        pattern_x = torch.sin(rot_x * config.frequency * math.pi * 2)
        pattern_y = torch.sin(rot_y * config.frequency * math.pi * 2)
        pattern = (pattern_x * pattern_y * 0.5 + 0.5)
    elif config.dot_type == "line":
        pattern = (torch.sin(rot_x * config.frequency * math.pi * 2) * 0.5 + 0.5)
    else:  # crosshatch
        line1 = torch.sin(rot_x * config.frequency * math.pi * 2)
        line2 = torch.sin(rot_y * config.frequency * math.pi * 2)
        pattern = (torch.maximum(line1, line2) * 0.5 + 0.5)

    return pattern.clamp(0, 1)


def apply_advanced_screentone(
    luma: torch.Tensor,
    config: ScreentoneConfig | None = None,
) -> torch.Tensor:
    """Applies halftone screentones to a grayscale luma image (H, W). Returns (3, H, W)."""
    if luma.ndim != 2:
        raise ValueError("luma tensor must be 2D (H, W).")

    config = config or ScreentoneConfig()
    pattern = generate_screentone_pattern(luma.shape, config, device=luma.device)

    # Convert luma into screentone threshold
    # Highlights (> 0.8) -> White
    # Shadows (< 0.2) -> Solid black
    # Midtones (0.2 .. 0.8) -> Screentone dots
    screentone_result = torch.ones_like(luma)
    midtone_mask = (luma >= 0.2) & (luma < 0.8)

    normalized_mid = (luma[midtone_mask] - 0.2) / 0.6
    screentone_result[midtone_mask] = (pattern[midtone_mask] < normalized_mid).float()
    screentone_result[luma < 0.2] = 0.0
    screentone_result[luma >= 0.8] = 1.0

    return screentone_result.repeat(3, 1, 1)
