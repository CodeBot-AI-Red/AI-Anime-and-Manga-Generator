"""Representação estruturada de personagem e cena."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .config import ModelConfig
from .tensors import Tensor


@dataclass(frozen=True)
class SceneDescription:
    """Controles de alto nível que serão condicionadores da futura geração."""

    character_id: str = ""
    pose: str = ""
    camera: str = ""
    background: str = ""
    art_style: str = "anime"

    def as_text(self) -> str:
        return "|".join((self.character_id, self.pose, self.camera, self.background, self.art_style))


class SceneConditioner:
    """Transforma metadados de cena em embeddings leves e determinísticos."""

    def __init__(self, config: ModelConfig) -> None:
        self.embedding_dim = config.scene_embedding_dim

    def encode(self, scenes: Sequence[SceneDescription]) -> Tensor:
        if not scenes:
            raise ValueError("Forneça ao menos uma descrição de cena.")
        if any(not isinstance(scene, SceneDescription) for scene in scenes):
            raise TypeError("Cada cena deve ser uma SceneDescription.")
        values = tuple(
            float((sum(ord(character) for character in scene.as_text()) + index) % 257) / 256
            for scene in scenes
            for index in range(self.embedding_dim)
        )
        return Tensor(shape=(len(scenes), self.embedding_dim), values=values)
