"""Pré-processamento local de amostras validadas para treinamento futuro.

O módulo não depende de frameworks de aprendizado de máquina. Atualmente ele
decodifica PNGs RGB/RGBA não entrelaçados, redimensiona por vizinho mais próximo
e produz o ``Tensor`` leve do núcleo inicial, no formato ``(C, H, W)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable
import zlib

from src.models.tensors import Tensor

from .dataset import DatasetSample
from .schema import ImageMetadata


class PreprocessingError(ValueError):
    """Indica imagem, dimensão ou dados inválidos no pré-processamento."""


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuração pequena para preparação local de imagens.

    ``resolution`` e ``batch_size`` possuem valores reduzidos para que os
    testes locais não exijam muitos recursos.
    """

    resolution: tuple[int, int] = (32, 32)
    batch_size: int = 2

    def __post_init__(self) -> None:
        if len(self.resolution) != 2 or any(dimension <= 0 for dimension in self.resolution):
            raise ValueError("resolution deve conter largura e altura positivas.")
        if self.batch_size <= 0:
            raise ValueError("batch_size deve ser positivo.")


@dataclass(frozen=True)
class ProcessedSample:
    """Imagem normalizada e seus metadados originais preservados."""

    tensor: Tensor
    metadata: ImageMetadata


@dataclass(frozen=True)
class ProcessedBatch:
    """Lote de tensores no formato ``(N, C, H, W)`` e metadados alinhados."""

    tensor: Tensor
    metadata: tuple[ImageMetadata, ...]


class DatasetPreprocessor:
    """Converte amostras do dataset em tensores normalizados sem treinar nada."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def process(self, sample: DatasetSample) -> ProcessedSample:
        """Processa uma única amostra e preserva exatamente seus metadados."""
        if sample is None or sample.image is None or sample.metadata is None:
            raise PreprocessingError("A amostra deve conter imagem e metadados.")
        pixels, width, height = self._read_png(sample.image.path)
        if sample.image.width is not None and sample.image.height is not None:
            if (width, height) != (sample.image.width, sample.image.height):
                raise PreprocessingError("As dimensões da imagem não correspondem aos metadados validados.")
        resized = self._resize(pixels, width, height, *self.config.resolution)
        return ProcessedSample(self._to_tensor(resized, *self.config.resolution), sample.metadata)

    def process_batch(self, samples: Iterable[DatasetSample]) -> ProcessedBatch:
        """Processa até ``batch_size`` amostras em um tensor único."""
        items = tuple(samples)
        if not items:
            raise PreprocessingError("Não é possível processar um lote vazio.")
        if len(items) > self.config.batch_size:
            raise PreprocessingError("O lote excede o batch_size configurado.")
        processed = tuple(self.process(sample) for sample in items)
        first_shape = processed[0].tensor.shape
        if any(item.tensor.shape != first_shape for item in processed):
            raise PreprocessingError("Os tensores do lote possuem dimensões incompatíveis.")
        values = tuple(value for item in processed for value in item.tensor.values)
        return ProcessedBatch(Tensor((len(processed),) + first_shape, values), tuple(item.metadata for item in processed))

    @staticmethod
    def _to_tensor(pixels: list[tuple[int, int, int]], width: int, height: int) -> Tensor:
        channels = tuple(
            pixel[channel] / 255.0
            for channel in range(3)
            for pixel in pixels
        )
        return Tensor((3, height, width), channels)

    @staticmethod
    def _resize(pixels: list[tuple[int, int, int]], source_width: int, source_height: int,
                target_width: int, target_height: int) -> list[tuple[int, int, int]]:
        return [
            pixels[(y * source_height // target_height) * source_width + (x * source_width // target_width)]
            for y in range(target_height)
            for x in range(target_width)
        ]

    @staticmethod
    def _read_png(path: object) -> tuple[list[tuple[int, int, int]], int, int]:
        try:
            data = path.read_bytes()  # type: ignore[union-attr]
        except (AttributeError, OSError) as error:
            raise PreprocessingError(f"Não foi possível ler a imagem: {path}") from error
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise PreprocessingError("O pré-processador atual requer uma imagem PNG válida.")

        width = height = color_type = bit_depth = None
        compressed = bytearray()
        position = 8
        try:
            while position < len(data):
                length = struct.unpack(">I", data[position:position + 4])[0]
                kind = data[position + 4:position + 8]
                chunk = data[position + 8:position + 8 + length]
                position += 12 + length
                if kind == b"IHDR":
                    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", chunk)
                    if bit_depth != 8 or color_type not in {2, 6} or compression or filtering or interlace:
                        raise PreprocessingError("PNG deve ser RGB/RGBA de 8 bits e não entrelaçado.")
                elif kind == b"IDAT":
                    compressed.extend(chunk)
                elif kind == b"IEND":
                    break
            if not width or not height or color_type is None or not compressed:
                raise PreprocessingError("PNG sem dados de imagem completos.")
            raw = zlib.decompress(compressed)
        except (struct.error, zlib.error, PreprocessingError) as error:
            if isinstance(error, PreprocessingError):
                raise
            raise PreprocessingError("PNG inválido ou corrompido.") from error

        channels = 4 if color_type == 6 else 3
        stride = width * channels
        expected = height * (stride + 1)
        if len(raw) != expected:
            raise PreprocessingError("Dados de pixel PNG possuem dimensões incompatíveis.")
        rows: list[bytes] = []
        offset = 0
        previous = bytes(stride)
        for _ in range(height):
            filter_type = raw[offset]
            encoded = raw[offset + 1:offset + 1 + stride]
            offset += stride + 1
            row = bytearray(encoded)
            for index in range(stride):
                left = row[index - channels] if index >= channels else 0
                above = previous[index]
                upper_left = previous[index - channels] if index >= channels else 0
                if filter_type == 1:
                    row[index] = (row[index] + left) & 255
                elif filter_type == 2:
                    row[index] = (row[index] + above) & 255
                elif filter_type == 3:
                    row[index] = (row[index] + ((left + above) // 2)) & 255
                elif filter_type == 4:
                    p = left + above - upper_left
                    pa, pb, pc = abs(p - left), abs(p - above), abs(p - upper_left)
                    row[index] = (row[index] + (left if pa <= pb and pa <= pc else above if pb <= pc else upper_left)) & 255
                elif filter_type != 0:
                    raise PreprocessingError("PNG usa um filtro de linha inválido.")
            previous = bytes(row)
            rows.append(previous)
        return [(row[index], row[index + 1], row[index + 2]) for row in rows for index in range(0, stride, channels)], width, height
