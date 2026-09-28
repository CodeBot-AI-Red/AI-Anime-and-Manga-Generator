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
    generation_steps: int = 4
    num_timesteps: int = 100
    time_embedding_dim: int = 32
    text_vocab_size: int = 256
    image_resolution: tuple[int, int] = (32, 32)

    def __post_init__(self) -> None:
        values = {
            "image_channels": self.image_channels,
            "base_channels": self.base_channels,
            "latent_channels": self.latent_channels,
            "text_embedding_dim": self.text_embedding_dim,
            "scene_embedding_dim": self.scene_embedding_dim,
            "downsample_factor": self.downsample_factor,
            "generation_steps": self.generation_steps,
            "num_timesteps": self.num_timesteps,
            "time_embedding_dim": self.time_embedding_dim,
            "text_vocab_size": self.text_vocab_size,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Valores de configuração devem ser positivos: {', '.join(invalid)}")
        if len(self.image_resolution) != 2 or any(dimension <= 0 for dimension in self.image_resolution):
            raise ValueError("image_resolution deve conter largura e altura positivas.")
        if any(dimension % self.downsample_factor for dimension in self.image_resolution):
            raise ValueError("image_resolution deve ser divisível por downsample_factor.")
