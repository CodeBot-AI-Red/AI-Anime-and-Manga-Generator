"""Pipeline local e modular de treinamento."""

from .config import TrainingConfig
from .data import ProcessedDataLoader, TrainingBatch
from .generative_model import GenerativeTrainingModel
from .loss import mean_squared_error
from .model import SmallReconstructionModel
from .trainer import Trainer

__all__ = ["DiffusionTrainingModel", "GenerativeTrainingModel", "ProcessedDataLoader", "SmallReconstructionModel", "Trainer", "TrainingBatch", "TrainingConfig", "mean_squared_error"]


def __getattr__(name: str):
    """Avoid importing the optional heavy training backend for legacy tools."""
    if name == "DiffusionTrainingModel":
        from .diffusion_model import DiffusionTrainingModel
        return DiffusionTrainingModel
    raise AttributeError(name)
