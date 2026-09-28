"""Loop modular de treinamento, validação, métricas e retomada."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .checkpoints import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .data import ProcessedDataLoader, TrainingBatch
from .loss import mean_squared_error
from .metrics import MetricHistory, mean_absolute_error
from .model import SmallReconstructionModel


class TrainableModel(Protocol):
    """Contrato mínimo compartilhado pelos modelos treináveis locais."""

    scale: float
    bias: float

    def forward(self, inputs: object, metadata: tuple[object, ...]) -> object: ...
    def backward(self, inputs: object, targets: object) -> dict[str, float]: ...
    def state_dict(self) -> dict[str, float]: ...
    def load_state_dict(self, state: dict[str, float]) -> None: ...


class Trainer:
    def __init__(self, config: TrainingConfig, model: TrainableModel | None = None) -> None:
        self.config, self.model = config, model or SmallReconstructionModel()
        self.optimizer_state = {"learning_rate": config.learning_rate}
        self.current_epoch = 0
        self.history = MetricHistory()

    def train_step(self, batch: TrainingBatch) -> float:
        predictions = self.model.forward(batch.inputs, batch.metadata)
        loss = mean_squared_error(predictions, batch.targets)
        gradients = self.model.backward(batch.inputs, batch.targets)
        self.model.scale -= self.config.learning_rate * gradients["scale"]
        self.model.bias -= self.config.learning_rate * gradients["bias"]
        return loss

    def validate(self, loader: ProcessedDataLoader) -> dict[str, float]:
        losses, maes = [], []
        for batch in loader.batches():
            predictions = self.model.forward(batch.inputs, batch.metadata)
            losses.append(mean_squared_error(predictions, batch.targets))
            maes.append(mean_absolute_error(predictions, batch.targets))
        return {"loss": sum(losses) / len(losses), "mae": sum(maes) / len(maes)}

    def fit(self, train_loader: ProcessedDataLoader, validation_loader: ProcessedDataLoader) -> MetricHistory:
        for epoch in range(self.current_epoch + 1, self.config.epochs + 1):
            losses = [self.train_step(batch) for batch in train_loader.batches(shuffle=True)]
            validation = self.validate(validation_loader)
            self.current_epoch = epoch
            self.history.train_loss.append(sum(losses) / len(losses))
            self.history.validation_loss.append(validation["loss"])
            self.history.validation_mae.append(validation["mae"])
            if epoch % self.config.save_every == 0:
                self.save_checkpoint()
        return self.history

    def save_checkpoint(self, path: Path | None = None) -> Path:
        target = path or self.config.checkpoint_dir / f"epoch-{self.current_epoch}.json"
        metrics = {"train_loss": self.history.train_loss[-1] if self.history.train_loss else 0.0}
        return save_checkpoint(target, self.current_epoch, self.model.state_dict(), self.optimizer_state, metrics)

    def resume(self, path: Path) -> None:
        checkpoint = load_checkpoint(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer_state = checkpoint["optimizer"]
        self.current_epoch = int(checkpoint["epoch"])
