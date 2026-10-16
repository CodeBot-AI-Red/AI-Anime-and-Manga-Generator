"""Manga layout, panel grids, gutters, and action speed lines generator.

Designed strictly for official Japanese manga page production layouts.
"""

from __future__ import annotations

import math
import torch


def apply_speed_lines(
    image: torch.Tensor,
    center: tuple[float, float] = (0.5, 0.5),
    line_density: float = 0.5,
    max_opacity: float = 0.85,
    inner_radius: float = 0.2,
) -> torch.Tensor:
    """Applies radial action speed lines radiating from a focus center onto a tensor (3, H, W)."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Image must have shape (3, H, W).")

    _, height, width = image.shape
    device = image.device

    ys = torch.linspace(0, 1, height, device=device)
    xs = torch.linspace(0, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    dx = grid_x - center[0]
    dy = grid_y - center[1]

    distance = torch.sqrt(dx.pow(2) + dy.pow(2))
    angle = torch.atan2(dy, dx)

    # High frequency angular sine waves for sharp speed line spikes
    frequency = 30.0 + line_density * 90.0
    pattern = (torch.sin(angle * frequency) * 0.5 + 0.5).pow(4)

    # Radial mask: inner radius clear, fading out towards edges
    radial_mask = ((distance - inner_radius) / (0.5 - inner_radius)).clamp(0, 1)

    speed_line_intensity = (pattern * radial_mask * max_opacity).clamp(0, 1)

    # Blend black speed lines into the image
    result = image * (1.0 - speed_line_intensity.unsqueeze(0))
    return result.clamp(0, 1)


class MangaPanelLayout:
    """Creates structured manga page layouts with gutters and frame borders."""

    @staticmethod
    def draw_grid_gutters(
        image: torch.Tensor,
        rows: int = 2,
        cols: int = 2,
        gutter_size: int = 4,
        border_size: int = 6,
        line_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> torch.Tensor:
        """Draws standard manga panel borders and gutters on a (3, H, W) tensor."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Image must have shape (3, H, W).")

        out = image.clone()
        _, height, width = out.shape

        color = torch.tensor(line_color, device=out.device, dtype=out.dtype)[:, None, None]

        # Draw outer borders
        out[:, :border_size, :] = color
        out[:, height - border_size :, :] = color
        out[:, :, :border_size] = color
        out[:, :, width - border_size :] = color

        # Draw row gutters
        for r in range(1, rows):
            y = int(height * r / rows)
            half_g = gutter_size // 2
            y_start = max(0, y - half_g)
            y_end = min(height, y + half_g + 1)
            out[:, y_start:y_end, :] = color

        # Draw column gutters
        for c in range(1, cols):
            x = int(width * c / cols)
            half_g = gutter_size // 2
            x_start = max(0, x - half_g)
            x_end = min(width, x + half_g + 1)
            out[:, :, x_start:x_end] = color

        return out
