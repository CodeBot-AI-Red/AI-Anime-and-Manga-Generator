"""Representação latente pequena e reversível para a geração experimental."""

from __future__ import annotations

from .config import ModelConfig
from .tensors import Tensor


class LatentRepresentation:
    """Codifica imagens BCHW por média espacial e decodifica por repetição.

    Esta implementação sem dependências fixa o contrato encoder/decoder. Redes
    variacionais ou autoencoders treináveis podem substituí-la mantendo a API.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def encode(self, images: Tensor) -> Tensor:
        if len(images.shape) != 4:
            raise ValueError("As imagens devem usar a forma (lote, canais, altura, largura).")
        batch, channels, height, width = images.shape
        if channels != self.config.image_channels:
            raise ValueError("Quantidade de canais de imagem incompatível.")
        factor = self.config.downsample_factor
        if height % factor or width % factor:
            raise ValueError("Altura e largura devem ser divisíveis pelo fator de redução.")
        latent_height, latent_width = height // factor, width // factor
        values = []
        image_size = channels * height * width
        for batch_index in range(batch):
            image = images.values[batch_index * image_size:(batch_index + 1) * image_size]
            for channel in range(self.config.latent_channels):
                source_channel = channel % channels
                for y in range(latent_height):
                    for x in range(latent_width):
                        pixels = [image[source_channel * height * width + (y * factor + dy) * width + x * factor + dx]
                                  for dy in range(factor) for dx in range(factor)]
                        values.append(sum(pixels) / len(pixels))
        return Tensor((batch, self.config.latent_channels, latent_height, latent_width), tuple(values))

    def decode(self, latents: Tensor) -> Tensor:
        if len(latents.shape) != 4 or latents.shape[1] != self.config.latent_channels:
            raise ValueError("Latentes devem usar a configuração do modelo.")
        batch, _, latent_height, latent_width = latents.shape
        factor, channels = self.config.downsample_factor, self.config.image_channels
        height, width = latent_height * factor, latent_width * factor
        values = []
        latent_size = self.config.latent_channels * latent_height * latent_width
        for batch_index in range(batch):
            latent = latents.values[batch_index * latent_size:(batch_index + 1) * latent_size]
            for channel in range(channels):
                offset = channel * latent_height * latent_width
                for y in range(height):
                    for x in range(width):
                        value = latent[offset + (y // factor) * latent_width + x // factor]
                        values.append(min(1.0, max(0.0, value)))
        return Tensor((batch, channels, height, width), tuple(values))
