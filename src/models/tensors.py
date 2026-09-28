"""Estruturas de tensor leves usadas pelo núcleo inicial.

O projeto ainda não depende de frameworks de aprendizado de máquina. Esta
pequena representação existe para tornar os contratos de forma explícitos e
para permitir que a arquitetura seja exercitada localmente. Ela pode ser
substituída por ``torch.Tensor`` quando o treinamento for introduzido.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable


@dataclass(frozen=True)
class Tensor:
    """Tensor denso mínimo, com dados achatados e dimensões verificáveis."""

    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("A forma do tensor deve conter apenas dimensões positivas.")
        if len(self.values) != prod(self.shape):
            raise ValueError("A quantidade de valores não corresponde à forma do tensor.")

    @classmethod
    def zeros(cls, shape: Iterable[int]) -> "Tensor":
        """Cria um tensor preenchido com zero para testes e prototipagem."""
        normalized_shape = tuple(shape)
        if not normalized_shape or any(dimension <= 0 for dimension in normalized_shape):
            raise ValueError("A forma do tensor deve conter apenas dimensões positivas.")
        return cls(shape=normalized_shape, values=(0.0,) * prod(normalized_shape))
