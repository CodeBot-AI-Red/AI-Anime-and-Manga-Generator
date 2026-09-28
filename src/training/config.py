"""Configuração validada do pipeline de treinamento local."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparâmetros pequenos e explícitos, seguros para execução local."""

    epochs: int = 2
    learning_rate: float = 0.01
    batch_size: int = 2
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 1
    device: str = "auto"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs deve ser positivo.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo.")
        if self.batch_size <= 0 or self.save_every <= 0:
            raise ValueError("batch_size e save_every devem ser positivos.")
        if self.device not in {"auto", "cpu"}:
            raise ValueError("device deve ser 'auto' ou 'cpu' nesta implementação local.")
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))

    @property
    def resolved_device(self) -> str:
        """Expõe o dispositivo realmente disponível sem depender de frameworks."""
        return "cpu"
