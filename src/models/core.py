"""Orquestração do núcleo inicial de IA de anime e mangá."""

from __future__ import annotations

from collections.abc import Sequence

from .config import ModelConfig
from .image_network import ImageEncoder
from .scene import SceneConditioner, SceneDescription
from .tensors import Tensor
from .text_conditioning import TextConditioner


class AnimeMangaCore:
    """Reúne imagem, texto e cena em uma saída latente validada."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.image_encoder = ImageEncoder(self.config)
        self.text_conditioner = TextConditioner(self.config)
        self.scene_conditioner = SceneConditioner(self.config)

    def forward(
        self,
        images: Tensor,
        prompts: Sequence[str],
        scenes: Sequence[SceneDescription],
    ) -> dict[str, Tensor]:
        """Valida um lote conjunto e retorna representações para etapas futuras."""
        image_latents = self.image_encoder.forward(images)
        text_embeddings = self.text_conditioner.encode(prompts)
        scene_embeddings = self.scene_conditioner.encode(scenes)
        batch_size = image_latents.shape[0]
        if text_embeddings.shape[0] != batch_size or scene_embeddings.shape[0] != batch_size:
            raise ValueError("Imagem, prompt e cena devem ter o mesmo tamanho de lote.")
        return {
            "image_latents": image_latents,
            "text_embeddings": text_embeddings,
            "scene_embeddings": scene_embeddings,
        }
