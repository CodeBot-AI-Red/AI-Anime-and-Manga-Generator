"""Montagem, validação e divisão determinística de datasets locais."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .images import ImageInfo, ImageLoader, ImageValidationError
from .metadata import MetadataReader
from .schema import ImageMetadata, MetadataValidationError


@dataclass(frozen=True)
class DatasetSample:
    image: ImageInfo
    metadata: ImageMetadata


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[DatasetSample, ...]
    validation: tuple[DatasetSample, ...]


class DatasetValidator:
    """Coleta erros por arquivo para que o usuário corrija o dataset local."""

    def validate(self, image_root: Path, annotation_root: Path) -> tuple[list[DatasetSample], list[str]]:
        loader = ImageLoader()
        reader = MetadataReader()
        samples: list[DatasetSample] = []
        errors: list[str] = []
        for image_path in loader.discover(image_root):
            try:
                samples.append(DatasetSample(loader.inspect(image_path), reader.read(image_path, image_root, annotation_root)))
            except (ImageValidationError, MetadataValidationError, ValueError) as error:
                errors.append(str(error))
        return samples, errors


def split_dataset(samples: list[DatasetSample], validation_fraction: float = 0.1, seed: str = "dataset-v1") -> DatasetSplit:
    """Divide exemplos por hash estável, preservando o resultado entre execuções."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction deve estar entre 0 (inclusivo) e 1 (exclusivo).")
    ordered = sorted(samples, key=lambda sample: sha256(f"{seed}:{sample.image.path}".encode()).hexdigest())
    validation_count = int(len(ordered) * validation_fraction)
    if validation_fraction > 0 and len(ordered) > 1:
        validation_count = max(1, validation_count)
    return DatasetSplit(tuple(ordered[validation_count:]), tuple(ordered[:validation_count]))
