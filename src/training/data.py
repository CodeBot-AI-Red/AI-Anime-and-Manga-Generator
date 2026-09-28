"""Adaptadores entre o pré-processamento existente e o treinamento."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Iterator

from src.data import ProcessedBatch, ProcessedSample
from src.models import Tensor


@dataclass(frozen=True)
class TrainingBatch:
    """Lote de entrada e alvo para a pequena tarefa de reconstrução local."""

    inputs: Tensor
    targets: Tensor
    metadata: tuple[object, ...]


class ProcessedDataLoader:
    """Agrupa ``ProcessedSample`` já produzidos pelo pré-processador do projeto."""

    def __init__(self, samples: Iterable[ProcessedSample], batch_size: int, seed: int = 42) -> None:
        self.samples = tuple(samples)
        if not self.samples:
            raise ValueError("O conjunto pré-processado não pode estar vazio.")
        if batch_size <= 0:
            raise ValueError("batch_size deve ser positivo.")
        self.batch_size, self.seed = batch_size, seed

    def batches(self, shuffle: bool = False) -> Iterator[TrainingBatch]:
        indices = list(range(len(self.samples)))
        if shuffle:
            Random(self.seed).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            items = [self.samples[index] for index in indices[start:start + self.batch_size]]
            yield self._make_batch(items)

    @staticmethod
    def _make_batch(items: list[ProcessedSample]) -> TrainingBatch:
        shape = items[0].tensor.shape
        if any(item.tensor.shape != shape for item in items):
            raise ValueError("Todas as amostras de um lote devem possuir a mesma forma.")
        tensor = Tensor((len(items),) + shape, tuple(value for item in items for value in item.tensor.values))
        return TrainingBatch(tensor, tensor, tuple(item.metadata for item in items))

    @classmethod
    def from_processed_batches(cls, batches: Iterable[ProcessedBatch], batch_size: int, seed: int = 42) -> "ProcessedDataLoader":
        """Permite consumir diretamente lotes retornados por ``DatasetPreprocessor``."""
        samples = []
        for batch in batches:
            channels, height, width = batch.tensor.shape[1:]
            size = channels * height * width
            samples.extend(
                ProcessedSample(Tensor((channels, height, width), batch.tensor.values[index * size:(index + 1) * size]), metadata)
                for index, metadata in enumerate(batch.metadata)
            )
        return cls(samples, batch_size, seed)
