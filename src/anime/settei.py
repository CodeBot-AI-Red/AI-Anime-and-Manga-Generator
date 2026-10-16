"""Anime production settei character sheets and model sheet generators.

Follows official studio production reference guidelines (character turnarounds,
expression guides, color swatches).
"""

from __future__ import annotations

import torch


class SetteiSheetGenerator:
    """Creates anime studio production settei model sheets with color swatches."""

    @staticmethod
    def create_settei_layout(
        character_image: torch.Tensor,
        title: str = "CHARACTER SETTEI MODEL SHEET",
        swatches: list[tuple[float, float, float]] | None = None,
    ) -> torch.Tensor:
        """Embeds a character image into a professional anime settei studio sheet layout."""
        if character_image.ndim != 3 or character_image.shape[0] != 3:
            raise ValueError("character_image must be a (3, H, W) tensor.")

        device = character_image.device
        dtype = character_image.dtype
        _, height, width = character_image.shape

        # Create studio sheet canvas (white background with bottom production bar)
        banner_height = max(32, height // 8)
        total_height = height + banner_height
        sheet = torch.ones((3, total_height, width), device=device, dtype=dtype)

        # Place character image in main area
        sheet[:, :height, :] = character_image

        # Production bottom bar line
        sheet[:, height : height + 2, :] = 0.2  # Dark line

        # Add color swatches at bottom right if provided
        swatches = swatches or [
            (0.96, 0.80, 0.70),  # Skin base
            (0.83, 0.63, 0.63),  # Skin shadow
            (0.75, 0.19, 0.25),  # Hair base
            (0.41, 0.06, 0.13),  # Hair shadow
            (0.12, 0.38, 0.80),  # Eye iris
        ]

        swatch_width = max(12, width // (len(swatches) * 2))
        swatch_height = max(12, banner_height - 8)

        for i, color in enumerate(swatches):
            x_start = width - (len(swatches) - i) * (swatch_width + 4) - 8
            x_end = x_start + swatch_width
            y_start = height + 4
            y_end = y_start + swatch_height

            if 0 <= x_start < width and 0 <= x_end <= width and y_end <= total_height:
                color_tensor = torch.tensor(color, device=device, dtype=dtype)[:, None, None]
                sheet[:, y_start:y_end, x_start:x_end] = color_tensor

        return sheet
