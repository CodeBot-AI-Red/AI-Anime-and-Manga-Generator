"""Leitura/escrita local de imagens como tensores ``(C, H, W)`` em [0, 1]."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import torch

from src.data.preprocessing import DatasetPreprocessor


def load_image(path: Path | str) -> torch.Tensor:
    """Lê PNG RGB/RGBA de 8 bits sem dependências; com Pillow instalado, aceita JPEG/WebP etc."""
    path = Path(path)
    try:
        from PIL import Image
    except ImportError:
        pixels, width, height = DatasetPreprocessor._read_png(path)
        return torch.tensor(pixels, dtype=torch.float32).view(height, width, 3).permute(2, 0, 1) / 255
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        data = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        return data.view(rgb.height, rgb.width, 3).permute(2, 0, 1).float() / 255


def load_images(paths: list[Path | str], size: tuple[int, int] | None = (224, 224)) -> torch.Tensor:
    """Empilha várias imagens em (N, 3, H, W), redimensionando para ``size`` (largura, altura)."""
    tensors = []
    for path in paths:
        image = load_image(path).unsqueeze(0)
        if size is not None:
            image = torch.nn.functional.interpolate(image, size=(size[1], size[0]), mode="bilinear", align_corners=False)
        tensors.append(image[0])
    return torch.stack(tensors)


def encode_png_bytes(image: torch.Tensor) -> bytes:
    """Converte ``(3, H, W)`` em [0, 1] para bytes de formato PNG de 8 bits sem dependência do Pillow."""
    pixels = bytes(image.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).contiguous().flatten().tolist())
    height, width = image.shape[1:]
    raw = b"".join(b"\x00" + pixels[row * width * 3:(row + 1) * width * 3] for row in range(height))

    def chunk(kind: bytes, content: bytes) -> bytes:
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content))

    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    )


def save_png(image: torch.Tensor, path: Path | str) -> Path:
    """Grava ``(3, H, W)`` em [0, 1] como PNG RGB sem Pillow."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png_bytes(image))
    return path
