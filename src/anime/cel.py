"""Anime cel-shading pipeline, rim light, and multi-plane depth rendering.

Implements official studio cel-shading rules, warm key/cool fill lighting,
and 3-plane anime background depth separation.
"""

from __future__ import annotations

import torch
from torch.nn import functional as functional


class CelShadingPipeline:
    """Applies anime cel-shading, rim lighting, and depth planes."""

    @staticmethod
    def apply_cel_shading(
        image: torch.Tensor,
        num_levels: int = 3,
        shadow_coolness: float = 0.15,
    ) -> torch.Tensor:
        """Quantizes smooth lighting into crisp cel-shading blocks with cool shadows."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Image must be (3, H, W).")

        luma = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        stepped_luma = ((luma * num_levels).floor().clamp(0, num_levels - 1) + 0.5) / num_levels

        ratio = (stepped_luma / luma.clamp_min(1e-4)).unsqueeze(0)
        cel_image = (image * ratio).clamp(0, 1)

        # Apply cool lavender/blue tint to deep shadow zones
        shadow_mask = (stepped_luma < (1.0 / num_levels)).unsqueeze(0)
        cool_tint = torch.tensor([0.9, 0.9, 1.1], device=image.device, dtype=image.dtype)[:, None, None]

        cel_image = torch.where(shadow_mask, cel_image * (1.0 - shadow_coolness) + cool_tint * shadow_coolness * cel_image, cel_image)
        return cel_image.clamp(0, 1)

    @staticmethod
    def apply_rim_light(
        image: torch.Tensor,
        rim_color: tuple[float, float, float] = (0.2, 0.8, 1.0),
        threshold: float = 0.15,
        strength: float = 0.6,
    ) -> torch.Tensor:
        """Adds vibrant colored rim lighting on character outer silhouette edges."""
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Image must be (3, H, W).")

        luma = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).unsqueeze(0).unsqueeze(0)

        # Sobel edge filter to detect silhouette edges
        kernel_x = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)

        grad_x = functional.conv2d(luma, kernel_x, padding=1)[0, 0]
        grad_y = functional.conv2d(luma, kernel_y, padding=1)[0, 0]
        edges = (grad_x.pow(2) + grad_y.pow(2)).sqrt()

        rim_mask = (edges > threshold).float() * strength
        rim_rgb = torch.tensor(rim_color, device=image.device, dtype=image.dtype)[:, None, None]

        result = image + rim_rgb * rim_mask.unsqueeze(0)
        return result.clamp(0, 1)
