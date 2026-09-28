"""Componentes locais de preparação de datasets de anime e mangá."""

from .dataset import DatasetSample, DatasetSplit, DatasetValidator, split_dataset
from .images import ImageInfo, ImageLoader, ImageValidationError
from .metadata import MetadataReader
from .preprocessing import DatasetPreprocessor, PreprocessingConfig, PreprocessingError, ProcessedBatch, ProcessedSample
from .schema import ImageMetadata, MetadataValidationError

__all__ = [
    "DatasetSample", "DatasetSplit", "DatasetValidator", "ImageInfo", "ImageLoader",
    "DatasetPreprocessor", "ImageMetadata", "ImageValidationError", "MetadataReader", "MetadataValidationError",
    "PreprocessingConfig", "PreprocessingError", "ProcessedBatch", "ProcessedSample", "split_dataset",
]
