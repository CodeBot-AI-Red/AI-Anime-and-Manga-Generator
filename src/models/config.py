"""Configuração pequena e validada para o núcleo de modelos."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Hiperparâmetros leves, adequados apenas à primeira versão local."""

    image_channels: int = 3
    base_channels: int = 8
    latent_channels: int = 16
    text_embedding_dim: int = 12
    scene_embedding_dim: int = 12
    downsample_factor: int = 2

    def __post_init__(self) -> None:
        values = {
            "image_channels": self.image_channels,
            "base_channels": self.base_channels,
            "latent_channels": self.latent_channels,
            "text_embedding_dim": self.text_embedding_dim,
            "scene_embedding_dim": self.scene_embedding_dim,
            "downsample_factor": self.downsample_factor,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Valores de configuração devem ser positivos: {', '.join(invalid)}")
