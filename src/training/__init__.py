"""Pipeline local e modular de treinamento."""

from .config import TrainingConfig
from .data import ProcessedDataLoader, TrainingBatch
from .loss import mean_squared_error
from .model import SmallReconstructionModel
from .trainer import Trainer

__all__ = ["ProcessedDataLoader", "SmallReconstructionModel", "Trainer", "TrainingBatch", "TrainingConfig", "mean_squared_error"]
