"""Configuração validada do núcleo de modelos.

O padrão agora é o perfil de **grande porte** (``ModelConfig()`` ==
``ModelConfig.large()``). O perfil antigo, pequeno e rápido para CPU, continua
disponível como ``ModelConfig.lite()`` para testes e experimentos locais.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ModelConfig:
    """Hiperparâmetros da U-Net de difusão, do encoder de texto e do encoder visual."""

    # --- Imagem e U-Net -------------------------------------------------
    image_channels: int = 3
    image_resolution: tuple[int, int] = (256, 256)
    base_channels: int = 128
    unet_res_blocks: int = 3
    downsample_factor: int = 2
    latent_channels: int = 16
    # --- Difusão ----------------------------------------------------------
    num_timesteps: int = 1000
    generation_steps: int = 50
    time_embedding_dim: int = 768
    condition_dropout: float = 0.1
    guidance_scale: float = 7.5
    # --- Encoder de texto (Transformer byte-level) ------------------------
    text_vocab_size: int = 256
    max_prompt_length: int = 512
    text_encoder_layers: int = 12
    attention_heads: int = 16
    # --- Visão: encoder de imagens (geradas ou enviadas pelo usuário) -----
    vision_enabled: bool = True
    vision_patch_size: int = 16
    vision_layers: int = 8
    vision_input_resolution: int = 224
    max_reference_images: int = 4
    reference_dropout: float = 0.5
    # --- Núcleo leve legado (Tensor puro em Python) ------------------------
    text_embedding_dim: int = 12
    scene_embedding_dim: int = 12
    generation_profile: str = "large"

    @classmethod
    def large(cls, **overrides: object) -> "ModelConfig":
        """Perfil de grande porte (padrão): 256×256, GPU recomendada."""
        return cls(**overrides)  # type: ignore[arg-type]

    @classmethod
    def xlarge(cls, **overrides: object) -> "ModelConfig":
        """Perfil de ultra grande porte: 512×512, 16 cabeças de atenção, 12 camadas de visão."""
        values: dict[str, object] = dict(
            image_resolution=(512, 512), base_channels=192, unet_res_blocks=3, num_timesteps=1000,
            generation_steps=50, time_embedding_dim=1024, guidance_scale=8.5, max_prompt_length=512,
            text_encoder_layers=16, attention_heads=16, vision_patch_size=16, vision_layers=12,
            vision_input_resolution=256, generation_profile="xlarge",
        )
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    @classmethod
    def xxlarge(cls, **overrides: object) -> "ModelConfig":
        """Perfil de estúdio supremo para grande porte: 1024×1024, 1.2B+ parâmetros de produção."""
        values: dict[str, object] = dict(
            image_resolution=(1024, 1024), base_channels=256, unet_res_blocks=4, num_timesteps=1000,
            generation_steps=50, time_embedding_dim=1280, guidance_scale=9.0, max_prompt_length=512,
            text_encoder_layers=16, attention_heads=16, vision_patch_size=16, vision_layers=16,
            vision_input_resolution=384, generation_profile="xxlarge",
        )
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    @classmethod
    def lite(cls, **overrides: object) -> "ModelConfig":
        """Perfil leve original (32×32) para CPU e testes."""
        values: dict[str, object] = dict(
            image_resolution=(32, 32), base_channels=16, unet_res_blocks=2, num_timesteps=100,
            generation_steps=4, time_embedding_dim=64, guidance_scale=3.0, max_prompt_length=128,
            text_encoder_layers=2, attention_heads=4, vision_patch_size=8, vision_layers=2,
            vision_input_resolution=32, generation_profile="lite",
        )
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "ModelConfig":
        """Recria a configuração salva em um checkpoint, ignorando chaves desconhecidas."""
        known = {field.name for field in fields(cls)}
        clean = {key: value for key, value in values.items() if key in known}
        if "image_resolution" in clean:
            clean["image_resolution"] = tuple(clean["image_resolution"])  # type: ignore[arg-type]
        return cls(**clean)  # type: ignore[arg-type]

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
            "vision_patch_size": self.vision_patch_size,
            "vision_layers": self.vision_layers,
            "vision_input_resolution": self.vision_input_resolution,
            "max_reference_images": self.max_reference_images,
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
        if self.vision_input_resolution % self.vision_patch_size:
            raise ValueError("vision_input_resolution deve ser divisível por vision_patch_size.")
        if not 0 <= self.condition_dropout < 1:
            raise ValueError("condition_dropout deve estar no intervalo [0, 1).")
        if not 0 <= self.reference_dropout <= 1:
            raise ValueError("reference_dropout deve estar no intervalo [0, 1].")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale não pode ser negativo.")
