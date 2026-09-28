"""Descoberta e validação leve de imagens locais, sem dependências externas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})


class ImageValidationError(ValueError):
    """Indica arquivo de imagem ausente, não suportado ou corrompido."""


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    format: str
    width: int | None
    height: int | None


class ImageLoader:
    """Lista imagens e confirma suas assinaturas antes de montar o dataset."""

    def discover(self, image_root: Path) -> list[Path]:
        if not image_root.is_dir():
            raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_root}")
        return sorted(
            (path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
            key=lambda path: path.as_posix(),
        )

    def inspect(self, path: Path) -> ImageInfo:
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ImageValidationError(f"Formato de imagem não suportado: {path}")
        try:
            header = path.read_bytes()[:32]
        except OSError as error:
            raise ImageValidationError(f"Não foi possível ler a imagem: {path}") from error

        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            return ImageInfo(path, "png", width, height)
        if header.startswith((b"GIF87a", b"GIF89a")) and len(header) >= 10:
            width, height = struct.unpack("<HH", header[6:10])
            return ImageInfo(path, "gif", width, height)
        if header.startswith(b"\xff\xd8"):
            return ImageInfo(path, "jpeg", *self._jpeg_dimensions(path))
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return ImageInfo(path, "webp", None, None)
        raise ImageValidationError(f"Conteúdo não reconhecido como imagem válida: {path}")

    @staticmethod
    def _jpeg_dimensions(path: Path) -> tuple[int | None, int | None]:
        data = path.read_bytes()
        position = 2
        while position + 9 <= len(data):
            if data[position] != 0xFF:
                position += 1
                continue
            marker = data[position + 1]
            position += 2
            if marker in {0xD8, 0xD9}:
                continue
            if position + 2 > len(data):
                break
            length = int.from_bytes(data[position:position + 2], "big")
            if length < 2 or position + length > len(data):
                break
            if 0xC0 <= marker <= 0xC3 and position + 7 <= len(data):
                return (
                    int.from_bytes(data[position + 5:position + 7], "big"),
                    int.from_bytes(data[position + 3:position + 5], "big"),
                )
            position += length
        raise ImageValidationError(f"JPEG sem dimensões válidas: {path}")
