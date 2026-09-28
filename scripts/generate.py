"""Sample a locally trained, from-scratch diffusion checkpoint to PNG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.config import ModelConfig
from src.training.diffusion_model import DiffusionTrainingModel


def save_png(image: torch.Tensor, path: Path) -> None:
    """Write RGB PNG without Pillow, keeping generation fully local."""
    import struct
    import zlib

    pixels = image.mul(255).round().byte().permute(1, 2, 0).contiguous().numpy().tobytes()
    height, width = image.shape[1:]
    raw = b"".join(b"\x00" + pixels[row * width * 3:(row + 1) * width * 3] for row in range(height))
    def chunk(kind: bytes, content: bytes) -> bytes:
        return struct.pack(">I", len(content)) + kind + content + struct.pack(">I", zlib.crc32(kind + content))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="guerreira anime em uma cidade ao entardecer")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("generated.png"))
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    model = DiffusionTrainingModel(ModelConfig(image_resolution=(args.resolution, args.resolution), generation_steps=args.steps))
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)
    image = model.sample([args.prompt], args.steps)[0]
    save_png(image, args.output)
    print(f"Geração concluída: {args.output} ({tuple(image.shape)})")


if __name__ == "__main__":
    main()
