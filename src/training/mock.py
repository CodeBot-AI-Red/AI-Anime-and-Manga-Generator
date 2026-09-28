"""Dados sintéticos determinísticos para testar o pipeline sem dataset real."""

from src.data import ImageMetadata, ProcessedSample
from src.models import Tensor


def synthetic_samples(count: int = 4, resolution: tuple[int, int] = (4, 4)) -> list[ProcessedSample]:
    if count <= 0:
        raise ValueError("count deve ser positivo.")
    width, height = resolution
    values_per_image = 3 * width * height
    return [ProcessedSample(Tensor((3, height, width), tuple(((index + offset) % 10) / 10 for offset in range(values_per_image))), ImageMetadata(caption=f"amostra sintética {index}")) for index in range(count)]
