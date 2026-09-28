"""Funções de perda sem dependência de framework externo."""

from src.models import Tensor


def mean_squared_error(predictions: Tensor, targets: Tensor) -> float:
    """Calcula MSE e rejeita tensores incompatíveis."""
    if predictions.shape != targets.shape:
        raise ValueError("Predições e alvos devem possuir a mesma forma.")
    return sum((prediction - target) ** 2 for prediction, target in zip(predictions.values, targets.values)) / len(predictions.values)
