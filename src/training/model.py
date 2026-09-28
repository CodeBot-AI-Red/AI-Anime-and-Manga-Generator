"""Modelo treinável mínimo, substituível por uma arquitetura maior no futuro."""

from __future__ import annotations

from src.models import AnimeMangaCore, SceneDescription, Tensor


class SmallReconstructionModel:
    """Reconstrói pixels com escala e viés treináveis, validando o núcleo existente."""

    def __init__(self) -> None:
        self.core = AnimeMangaCore()
        self.scale, self.bias = 0.5, 0.0

    def forward(self, inputs: Tensor, metadata: tuple[object, ...]) -> Tensor:
        prompts = [getattr(item, "caption", None) or "imagem de anime" for item in metadata]
        scenes = [SceneDescription(
            character_id=getattr(item, "character", None) or "", pose=getattr(item, "pose", None) or "",
            camera=getattr(item, "camera", None) or "", art_style=getattr(item, "style", None) or "anime",
        ) for item in metadata]
        self.core.forward(inputs, prompts, scenes)
        return Tensor(inputs.shape, tuple(self.scale * value + self.bias for value in inputs.values))

    def backward(self, inputs: Tensor, targets: Tensor) -> dict[str, float]:
        predictions = self.forward(inputs, (object(),) * inputs.shape[0])
        errors = [prediction - target for prediction, target in zip(predictions.values, targets.values)]
        coefficient = 2 / len(errors)
        return {"scale": coefficient * sum(error * value for error, value in zip(errors, inputs.values)), "bias": coefficient * sum(errors)}

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale, "bias": self.bias}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale, self.bias = float(state["scale"]), float(state["bias"])
