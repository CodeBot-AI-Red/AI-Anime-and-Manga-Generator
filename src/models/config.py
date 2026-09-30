"""Configuração pequena e validada para o núcleo de modelos."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Hiperparâmetros leves, adequados apenas à primeira versão local."""

    image_channels: int = 3
    base_channels: int = 16
    latent_channels: int = 16
    text_embedding_dim: int = 12
    scene_embedding_dim: int = 12
    downsample_factor: int = 2
    generation_steps: int = 4
    num_timesteps: int = 100
    time_embedding_dim: int = 64
    text_vocab_size: int = 256
    max_prompt_length: int = 128
    text_encoder_layers: int = 2
    attention_heads: int = 4
    unet_res_blocks: int = 2
    image_resolution: tuple[int, int] = (32, 32)
    condition_dropout: float = 0.1
    guidance_scale: float = 3.0

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
            "max_prompt_length": self.max_prompt_length,
            "text_encoder_layers": self.text_encoder_layers,
            "attention_heads": self.attention_heads,
            "unet_res_blocks": self.unet_res_blocks,
        }
        invalid = [name for name, value in values.items() if value <= 0]
        if invalid:
            raise ValueError(f"Valores de configuração devem ser positivos: {', '.join(invalid)}")
        if len(self.image_resolution) != 2 or any(dimension <= 0 for dimension in self.image_resolution):
            raise ValueError("image_resolution deve conter largura e altura positivas.")
        if any(dimension % self.downsample_factor for dimension in self.image_resolution):
            raise ValueError("image_resolution deve ser divisível por downsample_factor.")
        if self.time_embedding_dim % self.attention_heads:
            raise ValueError("time_embedding_dim deve ser divisível por attention_heads.")
        if not 0 <= self.condition_dropout < 1:
            raise ValueError("condition_dropout deve estar no intervalo [0, 1).")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale não pode ser negativo.")
