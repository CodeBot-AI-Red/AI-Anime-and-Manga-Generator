"""Métricas agregadas do treinamento."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.models import Tensor


def mean_absolute_error(predictions: Tensor, targets: Tensor) -> float:
    if predictions.shape != targets.shape:
        raise ValueError("Predições e alvos devem possuir a mesma forma.")
    return sum(abs(prediction - target) for prediction, target in zip(predictions.values, targets.values)) / len(predictions.values)


@dataclass
class MetricHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_mae: list[float] = field(default_factory=list)
